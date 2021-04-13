#!/bin/bash
git add .
echo "input commit reason..."
read reason
git commit -m "$reason"
git push origin master
