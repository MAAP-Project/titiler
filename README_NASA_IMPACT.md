# Titiler NASA IMPACT fork Readme

Clone the NASA-IMPACT fork of titiler and add a remote for the upstream "official" titiler repo. (Note: first check if upstream repo has migrated from 'master' to 'main')

```
git clone git@github.com:NASA-IMPACT/titiler.git
git remote add upstream git@github.com:developmentseed/titiler.git
git fetch upstream
git branch master --track origin/master
git checkout master
```


Determine which tag of titiler you want to merge in, then checkout it and merge.

```
export TITILER_VERSION=0.3.5 # whichever tag you want to merge from
git checkout -b merge-from-upstream-${TITILER_VERSION}
git merge tags/${TITILER_VERSION}
```

Resolve any conflicts from the merge. This is tricky!

Deploy it to a custom stage (e.g., `phil-merge-0.3.5`) and test.

Push the changes to the remote:

```
git push merge-from-upstream-${TITILER_VERSION} -u origin/merge-from-upstream-${TITILER_VERSION}
```

In GitHub, create a PR to merge the branch to main (previously master), get approvals, and merge with the "Merge commit" method (e.g., not squash).

