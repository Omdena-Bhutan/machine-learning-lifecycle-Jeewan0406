# Git commands
- To see which remote repository current local project is linked to
```
git remote -v

output:
If linked othewise will show nothing
origin  https://github.com/user/project.git (fetch)
origin  https://github.com/user/project.git (push)
```
- List of names of remote repository, nothing displayed if not configured.
```
git remote
```
- Saves the remote URL locally, as reference of remote repository URL 
```
git remote add origin https://github.com/user/project.git
```
- Actually connects and uploads the code
```
git push -u origin main
```
- Connects and downloads changes
```
git pull origin main
```
- Connects and check updates
```
git fetch origin
```
- Test connection to remote 
```
git ls-remote origin
```
- Remove remote repo
```
git remote remove origin
```
<hr style="height:4px;border:none;background-color:red;">

## DVC

```
(.venv) PS D:\MLOps\omdena-mlops-ml-lifecycle-new> dvc repro
dvc : The term 'dvc' is not recognized as the name of a cmdlet, function, script file, or operable program. Check the spelling of the name, 
or if a path was included, verify that the path is correct and try again.
At line:1 char:1
+ dvc repro
+ ~~~
    + CategoryInfo          : ObjectNotFound: (dvc:String) [], CommandNotFoundException
    + FullyQualifiedErrorId : CommandNotFoundException
```
- List down every package installed in the current environment and filter line containing word **dvc**. **|** sends the output to another command
```
pip list | findstr dvc
```
- Install dvc into current environment
```
pip install dvc
```
- Check version directly
```
dvc --version
```
- Remove the **dvc.lock** if there are old data and run dvc repro, builds lock form scratch
```
del dvc.lock
dvc repro
```
