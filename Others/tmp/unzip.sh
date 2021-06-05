#!/bin/bash
ls *.zip | while read line; do unzip $line; done
rm *.zip
