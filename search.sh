#!/bin/bash

echo "input search key words:"
read key_words
grep -l $key_words */*
