#!/bin/sh
t="_t"
for f in /Users/Shanil/final_proj/dataset_two_gas_sources_/dataset_twosources_downsampled/*
	do 
		k=$f$t
		echo "Processing $k file.." 
		sed 's/,/ /g' "$f" > "$k"
		cp "$k"  "$f"
		rm "$k"
done
i=0
for f in /Users/Shanil/final_proj/dataset_two_gas_sources_/dataset_twosources_downsampled/*
	do
		echo "$i"
		c=$(($i%5))
		if [ "$c" -eq "0" ]
			then
				k=$f
				echo "Main file : $k : is being appended to..."
			#	echo "$i"
				i=$(($i+1))
			#	echo "$i"
				continue
		fi
		cat $f>>$k
		rm $f
		i=$(($i+1))
done
