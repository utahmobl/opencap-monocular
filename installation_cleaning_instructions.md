Our installation guide is working well. We currently use the conda env 'opencap-mono' and the pipeline runs very well.

But i would like to simplify the installation for us and future users. I need to simplify it now so I can dockerize it and deploy it on many computers. Eventually I'll want it running on WSL as well but that's for later. Today I want a simplified guide and dockerize it. Let's start by creating a new conda env and trying to simplify from what we have. Do not mess with our current working env, create a new one and test it.

- removing the dependencies which are not needed such as for Slam etc (static camera only in our case)
- updating dependencies such as Pytorch etc
- anything else you have in mind?