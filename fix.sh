git filter-branch -f --env-filter '
if [ "$GIT_COMMITTER_NAME" = "Your Name" ];
then
    export GIT_COMMITTER_NAME="Baofeng Wang"
    export GIT_COMMITTER_EMAIL="brucecollege4success@gmail.com"
fi
if [ "$GIT_AUTHOR_NAME" = "Your Name" ];
then
    export GIT_AUTHOR_NAME="Baofeng Wang"
    export GIT_AUTHOR_EMAIL="brucecollege4success@gmail.com"
fi
' --tag-name-filter cat -- --branches --tags