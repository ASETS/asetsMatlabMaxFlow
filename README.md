## Advanced Segmentation Tools ([ASeTs](http://github.com/ASETS)) 
### Matlab repository ([website](http://asets.github.io/asetsMatlabMaxFlow/))

### License:  
TBD

### Developers:
- Martin Rajchl (@mrajchl), Imperial College London (UK)
- John SH. Baxter (@jshbaxter), Robarts Research Institute (CAN)
- Jing Yuan, Robarts Research Institute (CAN)

### Overview of file structure:   
*./* : Compile scripts, readme, license and todo list  
*./applications*: Contains examples of typical applications in image segmentation and analysis  
*./data*: Example data to run the applications  
*./lib*: Is created by compile.m and contains the compiled C/mex files  
*./maxflow*: Optimization code in C/mex and Matlab  
*./tests*: test scripts to compare different implementations against each other  

### Compile/Installation instructions:  
To compile the C/mex code run:
```matlab
compile.m
```
which creates the folder *./lib*. For testing purposes run any script in *./tests*.   

### Tests:  
- Matlab 2014a, 64-bit Linux (Ubuntu 12.04 LTS)  
- Matlab 201*, 64-bit Windows 7  

