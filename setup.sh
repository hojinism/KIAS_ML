#!/bin/bash
export WORKDIR=`pwd`

if [ -d "$WORKDIR/Particle_Images" ]
then
    echo data already exists
else
    mkdir -p $WORKDIR/Particle_Images/data
    wget https://cernbox.cern.ch/index.php/s/sHjzCNFTFxutYCj/download -O Particle_Images/data/SingleElectronPt50_IMGCROPS_n249k_RHv1.hdf5
    wget https://cernbox.cern.ch/index.php/s/69nGEZjOy3xGxBq/download -O Particle_Images/data/SinglePhotonPt50_IMGCROPS_n249k_RHv1.hdf5
fi
