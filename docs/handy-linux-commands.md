```bash
# View processes
ps -a

# Kill processes that may be hanging
sudo kill -9 <pids>
```

```bash
# Grab the question ids from a csv
# Skip header, keep first 1 (comma delimited), then write to ids.txt
tail -n+2 qa_pairs.csv | cut -d"," -f1 > ids.txt
```

```bash
# Make a copy of the first file, then append the second file (with header omitted) to the first
cp text_generations_part1.csv text_generations_full.csv && \
tail -n+2 text_generations_part2.csv >> text_generations_full.csv
```

```bash
# Few line watch of GPU usage
watch -n1 nvidia-smi --query-gpu=memory.used,memory.total,utilization.memory,utilization.gpu --format=csv,noheader
```
