#!/bin/bash
repository_ssh_url=git@github.com:mikaizhu/Notes.git
# we can use command : git remote -v to see what url is
git remote set-url origin $repository_ssh_url
git add .
echo "input commit reason..."
read reason
git commit -m "$reason"
git push origin master
