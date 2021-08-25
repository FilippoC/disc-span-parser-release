for i in ./*/;
do
	cd $i;
	sbatch cmd;
	cd ..;
done
