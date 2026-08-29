#! /bin/bash
directory="--directory=./"

# echo "==== Changes in number of communities ===="
# vertices=20000
# totalgraphs=11
# i=1
# for comm in 2 4 8 16 32 64 128 356 612 1024 2048 # 0.15 0.25 0.35 0.45 0.55 0.65 0.75 0.85 0.95
# do
#   echo "==== producing graph ${i} / ${totalgraphs} ===="
#   /usr/bin/python3 single_graph_generator.py -n ${vertices} -c ${comm} -o 2 -s 5
#   let i+=1
# done

for size in 40 50 60 70 ; do  # 25 50 75 100 ; do
  for overlap in 0.0 ; do  # 1.0 2.0 3.0 4.0 5.0 ; do
    for blocksizevar in 0.0 ; do # 1.0 2.0 3.0 4.0 5.0 ; do
      for biasvertices in 2 3 4 5 6 7 8 9 10 ; do
        communities=4
        biasblocks=4
        blockbias="0.8"
        vertexbias="1.0"
        python simple_generator.py -n $size -o $overlap -s $blocksizevar --biasblocks $biasblocks -b $blockbias --biasvertices $biasvertices --vertexbias $vertexbias $directory
#        python single_graph_generator.py -n $size -c -1 -e -1.0 -m 0.35 -a 0.25 -o $overlap -s $blocksizevar $directory
      done
    done
  done
done
