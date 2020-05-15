#!/bin/bash
for i in {0..10}
do
   python BO.py --bo_name qed_$i
done
