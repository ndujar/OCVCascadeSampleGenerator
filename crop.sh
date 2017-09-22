for file in *.ppm;
do
   convert $file -crop 100x40+0+20 $file
done
