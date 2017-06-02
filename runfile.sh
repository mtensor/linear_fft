python fouriernetwork.py > printout.txt
for i in {i..10}
do 
	python fouriernetwork.py >> printout.txt
	echo done run number $i
done	
