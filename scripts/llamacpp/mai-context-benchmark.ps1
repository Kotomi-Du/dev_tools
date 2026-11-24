<#
.SYNOPSIS
Runs the MAI context benchmark on this Windows host and appends to the CSV.
.EXAMPLE
.\mai-context-benchmark.ps1 -SplitMode tensor
.EXAMPLE
.\mai-context-benchmark.ps1 -SplitMode layer -CsvPath .\layer-rerun.csv
.NOTES
Tensor mode tests without DFlash. Layer mode tests both with and without it.
Each case uses one 256-token and one 8192-token synthetic request, 64 output
tokens each. New rows contain the exact launch command, its environment flag,
and the completion text returned by the server, including line breaks.
#>
param(
    [ValidateSet('tensor','layer')] [string]$SplitMode = 'tensor',
    [string]$CsvPath = (Join-Path $PSScriptRoot 'mai-context-benchmark.csv')
)
$ErrorActionPreference = 'Stop'
Add-Type -AssemblyName System.Net.Http
$bin = 'C:\Users\gta\Documents\project\llama'
$models = 'C:\Users\gta\Documents\Models\mai-code\models'
$exe = Join-Path $bin 'llama-server.exe'
$target = Join-Path $models 'MAI-Code-1.1-Flash-Q2-XS.gguf'
$drafter = Join-Path $models 'dflash-v3-lc1.gguf'
$gpuIds = @('0002ce91','000333c0')
$total = (Get-CimInstance Win32_OperatingSystem).TotalVisibleMemorySize * 1024
$columns = 'context,dflash,flash_requested,flash_target,flash_draft,requested_prompt_tokens,status,prompt_tokens,generated_tokens,prefill_tps,decode_tps,prefill_ms,decode_ms,draft_tokens,draft_accepted,host_baseline_mib,host_peak_delta_mib,idle_gpu0_dedicated_mib,peak_gpu0_dedicated_mib,idle_gpu0_shared_mib,peak_gpu0_shared_mib,idle_gpu1_dedicated_mib,peak_gpu1_dedicated_mib,idle_gpu1_shared_mib,peak_gpu1_shared_mib,idle_host_used_mib,peak_host_used_mib,samples,detail,split_mode,command_line,command_source,env_ggml_test_mm_f32acc,generated_text' -split ','
$prose = @'
The following field notes describe a small town planning a community garden. Neighbors meet each week to compare soil moisture, count seedlings, and decide which paths need repairs. They record observations in a shared notebook. On a cold morning the group repaired two raised beds and moved the water barrels away from the old gate. When the rain stopped, volunteers measured each path and marked places where puddles formed. The coordinator prepared a short checklist for the next meeting. The purpose of these notes is to turn observations into practical recommendations that can be explained to new volunteers.
'@

function Server-Arguments($mode,$ctx,$flash,$draft,$port) {
    $flags = @('-m',"`"$target`"",'-ngl','999','-fa',$flash,'-np','1','--no-host','--fit','off','-sm',$mode,'-c',"$ctx",'-ctk','f16','-ctv','f16','-b','2048','-ub','1024','--ctx-checkpoints','0','--no-cache-prompt','--host','127.0.0.1','--port',"$port",'-lv','4')
    if ($draft) {
        $flags += @('-md',"`"$drafter`"",'--spec-type','draft-dflash','--spec-draft-ngl','all','--spec-draft-n-max','2','--spec-draft-p-min','0.2','-ctkd','f16','-ctvd','f16')
    } else { $flags += @('--spec-type','none') }
    return $flags
}

function Measure-Memory($serverId) {
    $samples = (Get-Counter '\GPU Adapter Memory(*)\Dedicated Usage','\GPU Process Memory(*)\Shared Usage','\Memory\Available Bytes' -ErrorAction SilentlyContinue).CounterSamples
    $mem = [ordered]@{}
    for ($i=0; $i -lt 2; $i++) {
        $key = "luid_0x00000000_0x$($gpuIds[$i])_phys_0"
        $ded = $samples | Where-Object { $_.InstanceName -eq $key -and $_.Path.EndsWith('\dedicated usage') } | Select-Object -First 1
        $shared = $samples | Where-Object { $_.InstanceName -eq "pid_${serverId}_$key" -and $_.Path.EndsWith('\shared usage') } | Select-Object -First 1
        $mem["gpu${i}_dedicated_mib"] = if ($ded) { [math]::Round($ded.CookedValue / 1MB,1) }
        $mem["gpu${i}_shared_mib"] = if ($shared) { [math]::Round($shared.CookedValue / 1MB,1) }
    }
    $avail = $samples | Where-Object { $_.Path.EndsWith('\memory\available bytes') } | Select-Object -First 1
    $mem['host_used_mib'] = if ($avail) { [math]::Round(($total-$avail.CookedValue)/1MB,1) }
    return $mem
}

function Post-Json($endpoint,$payload) {
    $json = ConvertTo-Json -Depth 5 -Compress -InputObject $payload
    return $http.PostAsync("$base/$endpoint",[Net.Http.StringContent]::new($json,[Text.Encoding]::UTF8,'application/json'))
}

function Read-Reply($pending) {
    $response = $pending.GetAwaiter().GetResult()
    try {
        $body = $response.Content.ReadAsStringAsync().GetAwaiter().GetResult()
        if (!$response.IsSuccessStatusCode) { throw "HTTP $([int]$response.StatusCode): $body" }
        return ConvertFrom-Json -InputObject $body
    } finally { $response.Dispose() }
}

if (Get-Process -Name llama-server -ErrorAction SilentlyContinue) { throw 'An existing llama-server is running' }
$CsvPath = $ExecutionContext.SessionState.Path.GetUnresolvedProviderPathFromPSPath($CsvPath)
if (Test-Path -LiteralPath $CsvPath) {
    $old = @(Import-Csv -LiteralPath $CsvPath)
    if (!$old.Count) { throw 'Expected an existing benchmark CSV with data rows' }
    $oldColumns = @($old[0].PSObject.Properties.Name)
    if (($oldColumns -join ',') -cne ($columns -join ',')) {
        $legacy = ($oldColumns -join ',') -ceq ($columns[0..28] -join ',')
        $withoutText = ($oldColumns -join ',') -ceq ($columns[0..32] -join ',')
        if (!$legacy -and !$withoutText) { throw 'Unexpected benchmark CSV schema' }
        $prior = foreach ($r in $old) {
            $fields = [ordered]@{}
            foreach ($name in $oldColumns) { $fields[$name] = $r.$name }
            if ($legacy) {
                $fields['split_mode'] = 'layer'
                $fields['command_line'] = '"' + $exe + '" ' + ((Server-Arguments 'layer' $r.context $r.flash_requested ($r.dflash -eq 'True') '<port>') -join ' ')
                $fields['command_source'] = 'reconstructed; original loopback port not recorded for all configurations'
                $fields['env_ggml_test_mm_f32acc'] = '1'
            }
            $fields['generated_text'] = $null
            [pscustomobject]$fields
        }
        $tempCsv = "$CsvPath.$([guid]::NewGuid()).tmp"
        $backupCsv = "$tempCsv.bak"
        try {
            $prior | Export-Csv -LiteralPath $tempCsv -NoTypeInformation -Encoding UTF8
            [IO.File]::Replace($tempCsv,$CsvPath,$backupCsv)
            Remove-Item -LiteralPath $backupCsv
        } finally { if (Test-Path -LiteralPath $tempCsv) { Remove-Item -LiteralPath $tempCsv } }
    }
}

$previousEnv = $env:GGML_TEST_MM_F32ACC
$env:GGML_TEST_MM_F32ACC = '1'
$handler = [Net.Http.HttpClientHandler]::new()
$handler.UseProxy = $false
$http = [Net.Http.HttpClient]::new($handler)
$http.Timeout = [TimeSpan]::FromMinutes(15)
$requests = @{}
$draftModes = @($false)
if ($SplitMode -eq 'layer') { $draftModes += $true }
try {
    foreach ($ctx in @(131072,262144)) {
        foreach ($flash in @('on','off')) {
            foreach ($draft in $draftModes) {
                $baseline = Measure-Memory 0
                $listener = [Net.Sockets.TcpListener]::new([Net.IPAddress]::Loopback,0)
                $listener.Start()
                try { $port = $listener.LocalEndpoint.Port } finally { $listener.Stop() }
                $base = "http://127.0.0.1:$port"
                $flags = Server-Arguments $SplitMode $ctx $flash $draft $port
                $command = '"' + $exe + '" ' + ($flags -join ' ')
                $stem = Join-Path $PSScriptRoot ('mai-run-'+[guid]::NewGuid())
                $out = "$stem.out"
                $err = "$stem.err"
                $server = $null
                $setupError = $null
                $status = 'startup failed'
                $idle = @{}
                $flashModes = @()
                Write-Host "Starting $SplitMode context=$ctx flash=$flash DFlash=$draft"
                try {
                    try {
                        $server = Start-Process -FilePath $exe -WorkingDirectory $bin -ArgumentList $flags -RedirectStandardOutput $out -RedirectStandardError $err -PassThru
                        $ready = $false
                        $deadline = (Get-Date).AddMinutes(5)
                        while ((Get-Date) -lt $deadline -and !$server.HasExited) {
                            $health = & curl.exe --noproxy '*' -fs --max-time 2 "$base/health"
                            if ($LASTEXITCODE -eq 0 -and $health -match '"status"\s*:\s*"ok"') { $ready = $true; break }
                            Start-Sleep -Seconds 2
                        }
                        if (!$ready) {
                            if ($server.HasExited) { throw "Server exited during startup ($($server.ExitCode))" }
                            throw 'Timed out waiting for startup'
                        }
                        $status = 'setup failed'
                        $log = Get-Content -LiteralPath $err
                        $flashModes = @($log | Select-String 'llama_context: flash_attn\s*=\s*(\S+)' | ForEach-Object { $_.Matches[0].Groups[1].Value })
                        $expectedFlash = if ($flash -eq 'on') { 'enabled' } else { 'disabled' }
                        $props = Read-Reply ($http.GetAsync("$base/props"))
                        if ($props.default_generation_settings.n_ctx -ne $ctx) { throw 'Server reported a different slot context' }
                        if (!$flashModes.Count -or $flashModes[0] -ne $expectedFlash) { throw 'Target Flash Attention mode does not match the request' }
                        if ($draft -and ($flashModes.Count -lt 2 -or $flashModes[1] -ne $expectedFlash -or !($log -match "adding speculative implementation 'draft-dflash'"))) { throw 'DFlash or its Flash Attention mode was not initialized as requested' }
                        $idle = Measure-Memory $server.Id
                        if (!$requests.Count) {
                            $intro = 'Write a detailed numbered checklist for the community garden based on these notes. Include practical details and explain why each step matters.'
                            $shortSrc = "$intro`n$prose`n$prose`n$prose"
                            $longSrc = "$intro`n" + (($prose + "`n") * 120)
                            $newRequests = @{}
                            foreach ($src in @(@{text=$shortSrc; count=256},@{text=$longSrc; count=8192})) {
                                $tok = Read-Reply (Post-Json 'tokenize' @{content=$src.text; add_special=$true})
                                if ($tok.tokens.Count -lt $src.count) { throw "Tokenization returned too few tokens for $($src.count)" }
                                $newRequests[$src.count] = @($tok.tokens[0..($src.count-1)])
                            }
                            $requests = $newRequests
                        }
                    } catch {
                        $setupError = $_.Exception.Message
                        if (Test-Path -LiteralPath $err) {
                            $explanation = @(Get-Content -LiteralPath $err -Tail 16 | Where-Object { $_ -match 'error|failed|not supported|not implemented|assert|requires|cannot|must|out of memory' } | Select-Object -Last 4)
                            if ($explanation.Count) { $setupError += '; ' + ($explanation -join '; ') }
                        }
                    }
                    foreach ($count in @(256,8192)) {
                        $row = [ordered]@{}
                        foreach ($name in $columns) { $row[$name] = $null }
                        $row.context = $ctx; $row.dflash = $draft; $row.flash_requested = $flash
                        if ($flashModes.Count) { $row.flash_target = $flashModes[0] }
                        if ($draft -and $flashModes.Count -gt 1) { $row.flash_draft = $flashModes[1] }
                        $row.requested_prompt_tokens = $count
                        $row.host_baseline_mib = $baseline['host_used_mib']
                        $row.samples = 0
                        $row.split_mode = $SplitMode; $row.command_line = $command
                        $row.command_source = 'recorded'; $row.env_ggml_test_mm_f32acc = '1'
                        foreach ($key in $idle.Keys) { $row["idle_$key"] = $idle[$key] }
                        if ($setupError) {
                            $row.status = $status; $row.detail = $setupError
                        } else {
                            try {
                                $pending = Post-Json 'completion' @{prompt=$requests[$count]; n_predict=64; temperature=0; seed=42; ignore_eos=$true; cache_prompt=$false; stream=$false}
                                do {
                                    $mem = Measure-Memory $server.Id
                                    foreach ($key in $mem.Keys) {
                                        $field = "peak_$key"
                                        if ($null -ne $mem[$key] -and ($null -eq $row[$field] -or $mem[$key] -gt $row[$field])) { $row[$field] = $mem[$key] }
                                    }
                                    $row.samples++
                                    if (!$pending.IsCompleted) { Start-Sleep -Seconds 1 }
                                } while (!$pending.IsCompleted)
                                $reply = Read-Reply $pending
                                if ($reply.content -isnot [string]) { throw 'Completion response did not contain a text string' }
                                $row.generated_text = $reply.content
                                $t = $reply.timings
                                $row.prompt_tokens = $t.prompt_n; $row.generated_tokens = $t.predicted_n
                                $row.prefill_tps = $t.prompt_per_second; $row.decode_tps = $t.predicted_per_second
                                $row.prefill_ms = $t.prompt_ms; $row.decode_ms = $t.predicted_ms
                                if ($draft) { $row.draft_tokens = $t.draft_n; $row.draft_accepted = $t.draft_n_accepted }
                                if ($t.prompt_n -ne $count -or $t.predicted_n -ne 64 -or $reply.truncated -or $t.prompt_per_second -le 0 -or $t.predicted_per_second -le 0) { throw 'Unexpected token count, timing, or truncation' }
                                if ($null -eq $row.peak_host_used_mib -or $null -eq $row.peak_gpu0_shared_mib -or $null -eq $row.peak_gpu1_shared_mib) { throw 'Required memory counters were unavailable' }
                                $row.status = 'ok'
                            } catch { $row.status = 'request failed'; $row.detail = $_.Exception.Message }
                            if ($null -ne $row.peak_host_used_mib -and $null -ne $row.host_baseline_mib) {
                                $row.host_peak_delta_mib = [math]::Round($row.peak_host_used_mib - $row.host_baseline_mib,1)
                            }
                        }
                        [pscustomobject]$row | Export-Csv -LiteralPath $CsvPath -NoTypeInformation -Append -Encoding UTF8
                        Write-Host "$count tokens : $($row.status) | prefill=$($row.prefill_tps) decode=$($row.decode_tps) | $($row.detail)"
                    }
                } finally {
                    if ($server -and !$server.HasExited) { Stop-Process -Id $server.Id; $null = $server.WaitForExit(15000) }
                    if ($server) { $server.Dispose() }
                    foreach ($f in @($out,$err)) { if (Test-Path -LiteralPath $f) { Remove-Item -LiteralPath $f -ErrorAction Continue } }
                    Start-Sleep -Seconds 3
                }
            }
        }
    }
} finally {
    $http.Dispose()
    $env:GGML_TEST_MM_F32ACC = $previousEnv
}
