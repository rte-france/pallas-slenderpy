# Build the documentation with Sphinx
## On Windows 10
If your Python distribution is Anaconda, you can build the documentation with the following steps:

1. Install Sphinx:<br>
	`conda install sphinx`
2. Add the variable SPHINXBUILD to your environment variables. Its value is the full path to sphinx-build.exe.
 It should be something like C:\Users\user_name\miniconda3\Scripts\sphinx-build.exe
3. Open a command line interpreter (cmd.exe) and run the batch file make.bat:<br>
	`make html`
	
## On Linux
1. Install Sphinx:<br>
	`sudo apt-get install python3-sphinx`
2. Open a terminal and run the Makefile:<br>
	`make html`
	
# Read the documentation
Once built following the steps given above, the documentation is in *doc/_build/html/*.
Just open the file *index.html *with your favorite Internet browser.
