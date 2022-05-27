echo Searching for best ckpt in log.txt
grep -B 1 "best" log.txt > best_tmp.txt
echo Filtering best ckpt indexes
grep "model-[0-9]*.pth" -o best_tmp.txt > best_models.txt
echo Deleting tmp file
rm best_tmp.txt
echo Complete
echo Press any key to quit
read -n 1