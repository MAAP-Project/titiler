# Titiler NASA IMPACT fork Readme

```
export TITILER_VERSION=0.3.5 # whichever tag you want to merge from
git clone git@github.com:NASA-IMPACT/titiler.git
git remote add upstream git@github.com:developmentseed/titiler.git
git fetch upstream
```

```
git branch master --track origin/master
git checkout master
git checkout -b merge-from-upstream-${TITILER_VERSION}
git merge tags/${TITILER_VERSION}
```

Resolve any conflicts from the merge.

Push to the remote:

```
git push merge-from-upstream-${TITILER_VERSION} -u origin/merge-from-upstream-${TITILER_VERSION}
```

In GitHub, create a PR to merge the branch to master, get approvals, and merge with the "Merge commit" method (e.g., not squash).

