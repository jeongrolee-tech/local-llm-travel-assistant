set -u
for m in "qwen3.5:4b" "qwen3.5:9b" "gemma4:e2b" "gemma4:e4b" "exaone3.5:7.8b" "hf.co/mradermacher/kanana-2-3b-instruct-GGUF:Q4_K_M"; do
  curl -s http://127.0.0.1:11434/api/generate -d "{\"model\":\"$m\",\"prompt\":\"hi\",\"stream\":false,\"options\":{\"num_predict\":1}}" > /dev/null
  echo "##### $m"
  curl -s http://127.0.0.1:11434/api/ps | python -c "
import sys,json
d=json.load(sys.stdin)
for x in d.get('models',[]):
    tot=x.get('size',0); vr=x.get('size_vram',0)
    print('  size_total_MiB=%.0f  size_vram_MiB=%.0f  gpu_pct=%.0f%%  ctx=%s  quant=%s' % (tot/1048576, vr/1048576, (vr/tot*100 if tot else 0), x.get('context_length'), x.get('details',{}).get('quantization_level')))
"
  curl -s http://127.0.0.1:11434/api/generate -d "{\"model\":\"$m\",\"keep_alive\":0}" > /dev/null
done
