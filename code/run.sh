max_iter=1000;
for dsi in 0 1
do
	for nl in 14 40 100
	do
		for l in 10 1 0.1 0.01 
		do
			./main $dsi $nl $max_iter $l > $dsi_$nl_$l.log;
		done
	done 
done 
