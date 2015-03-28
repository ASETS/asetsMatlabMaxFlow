## Advanced Segmentation Tools ([ASeTs](http://github.com/ASETS)) Matlab repository ([website](http://asets.github.io/asetsMatlabMaxFlow/))

### License:  
BSD (see license.md)

### Developers:
- Martin Rajchl (@mrajchl), Imperial College London (UK)
- John SH. Baxter (@jshbaxter), Robarts Research Institute (CAN)
- Jing Yuan, Robarts Research Institute (CAN)

### Features: 
- Fast parallel continuous max flow solvers in 2D/3D
    - Binary max flow
    - Multi-region (Potts model, Ishikawa model, Hierarchical Max Flow)
    - In two different implementations (full flow and pseudo flow solvers)
- Implemented in multiple languages
   - Matlab/mex/C
   - Matlab/CUDA
- Tutorials
   - T01 Binary Graph Cut
   - T02 Multi-region color image segmentation with the Potts model
   - T03 Using different max flow implementations (Matlab, C, CUDA) 
- Application examples for (medical) image segmentation:
    - Interactive max flow graph cuts
    - Regularization of probabilistic label maps as in atlas-based segmentation
    - High-performance multi-phase levelsets 
    - Post-processing of flawed manual segmentations with constrast sensitive regularization
    - L1 intensity segmentation  

### Overview of file structure:   
*./*: Compile scripts, readme, license and todo list  
*./applications*: Contains examples of typical applications in image segmentation and analysis  
*./data*: Example data to run the applications  
*./lib*: Is created by compile.m and contains the compiled C/mex files  
*./maxflow*: Optimization code in C/mex and Matlab  
*./tests*: Test scripts to compare different implementations against each other
*./tutorials*: Contains available tutorials   

### Compile/Installation instructions:  
To compile the C/mex code run:
```matlab
compile.m
```
which creates the folder *./lib*. For testing purposes run any script in *./tests*.   

### Tests:  
- Matlab 2014a, 64-bit Linux (Ubuntu 12.04 LTS)  
- Matlab 2015a, 64-bit Windows 7
- Matlab 2012a, 32-bit WinXP

