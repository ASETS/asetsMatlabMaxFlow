%   Martin Rajchl, Imperial College London, 2015
%   compile script for mex/c code solvers in asetsMaxFlow

%   compile the mex implementations before running
basePath = pwd();
libDir = [basePath, filesep, 'lib']; 
mkdir(libDir)

% compile
cd(libDir);
mex ([basePath, filesep, 'maxflow', filesep ,'asetsPotts2D_mex.c']);
mex ([basePath, filesep, 'maxflow', filesep ,'asetsPotts3D_mex.c']);
cd(basePath);
