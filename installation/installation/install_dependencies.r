
# This script manages the R Library installation for the libraries specified in r_requirements.txt

pkgs = read.csv('installation/r_requirements.txt', header=FALSE, stringsAsFactors=FALSE)$V1
install.packages(pkgs, repos='http://cloud.r-project.org/')
