for i in {100..1000}
do 
	echo $i $i $(($i/15)) 100
	./build/mmm $1 $i $i $(($i/15)) 100
done
