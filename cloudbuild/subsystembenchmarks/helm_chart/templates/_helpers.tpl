{{- define "ssb.validate" -}}
{{- $acc := toString .Values.accelerator -}}
{{- if not (has $acc (list "gpu" "tpu")) -}}
  {{- fail (printf "accelerator must be 'gpu' or 'tpu', got %q" $acc) -}}
{{- end -}}
{{- $nodes := int .Values.nodes -}}
{{- if lt $nodes 1 -}}
  {{- fail (printf "nodes must be >= 1, got %d" $nodes) -}}
{{- end -}}
{{- if eq $acc "tpu" -}}
  {{- if gt (len .Release.Name) 20 -}}
    {{- fail (printf "TPU JobSet release name must be <= 20 characters (got %d: %q)" (len .Release.Name) .Release.Name) -}}
  {{- end -}}
  {{- if not .Values.tpu.accelerator -}}
    {{- fail "tpu.accelerator must be non-empty when accelerator=tpu" -}}
  {{- end -}}
  {{- $chips := int .Values.tpu.chipsPerNode -}}
  {{- if lt $chips 1 -}}
    {{- fail (printf "tpu.chipsPerNode must be >= 1, got %d" $chips) -}}
  {{- end -}}
  {{- $prio := toString .Values.tpu.priority -}}
  {{- if not (has $prio (list "medium" "low" "very-low")) -}}
    {{- fail (printf "tpu.priority must be one of medium, low, very-low (got %q)" $prio) -}}
  {{- end -}}
  {{- $topo := index .Values.tpu.topology (toString $nodes) -}}
  {{- if not $topo -}}
    {{- fail (printf "no tpu.topology entry configured for nodes=%d" $nodes) -}}
  {{- end -}}
  {{- if and (gt $nodes 1) (not .Values.tpu.multiNodePlacementPolicy) -}}
    {{- fail "tpu.multiNodePlacementPolicy must be set when accelerator=tpu and nodes > 1" -}}
  {{- end -}}
{{- end -}}
{{- end -}}

{{- define "ssb.image" -}}
{{- if eq (toString .Values.accelerator) "tpu" -}}
  {{- .Values.image.tpu -}}
{{- else -}}
  {{- .Values.image.gpu -}}
{{- end -}}
{{- end -}}

{{- define "ssb.machineTypeDefault" -}}
{{- if eq (toString .Values.accelerator) "tpu" -}}
  {{- .Values.tpu.machineType -}}
{{- else -}}
  {{- .Values.gpu.machineType -}}
{{- end -}}
{{- end -}}

{{- define "ssb.tpuTopology" -}}
{{- index .Values.tpu.topology (toString (int .Values.nodes)) -}}
{{- end -}}

{{- define "ssb.buildId" -}}
{{- .Values.buildId | default .Values.runId -}}
{{- end -}}

{{- define "ssb.shouldPinZone" -}}
{{- $bt := toString .Values.bucketType -}}
{{- if or .Values.pinZone (eq $bt "zonal") (hasPrefix "rapid_cache_" $bt) (contains "_rcu_" $bt) -}}
true
{{- end -}}
{{- end -}}

{{- define "ssb.leaderAddr" -}}
{{- printf "%s-workload-0-0.%s.%s.svc.cluster.local" .Release.Name .Release.Name .Release.Namespace -}}
{{- end -}}
