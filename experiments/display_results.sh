for i in ./*/;
do
	echo $i;
	for j in $i/*_all.txt;
	do
		echo -n $j;
		echo -n "     ";
		cat $j | grep f-measure | awk '{print $4}'
	done;
	echo;
	for j in $i/*_disc.txt;
	do
		echo -n $j;
		echo -n "     ";
		cat $j | grep f-measure | awk '{print $4}'
	done;
	echo;
	echo "---";
	echo;
done;
