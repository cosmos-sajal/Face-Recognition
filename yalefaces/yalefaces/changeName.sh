#!/bin/bash

for name in `ls ./ | grep -v .gif`; do
    new_name=$name".gif"
    mv $name $new_name
done