#!/bin/bash
git add .
read -p "input commit reason... " reason
git commit -m "$reason"
git push origin master
