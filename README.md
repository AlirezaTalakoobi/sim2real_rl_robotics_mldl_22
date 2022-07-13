# Project5_Group5 Submission

## Group Members

- Tommaso Massaglia, s292988@studenti.polito.it
- Farzad Imanpour Sardroudi, s289265@studenti.polito.it
- Alireza Talakoobi, s289641@studenti.polito.it

## Repo structure

Each of the implemented models or domain randomization algorithm can be found in its own folder, along with the required files to run both training and testing as well as one of the produced models that we used for testing in the paper.

The folder results_plots contains several .ipynb files that were used to generate the plots used in the paper as well as the figures themselves.

Of note, to implement UDR we edited the env/custom_hopper.py itself so that whenever the gym.make method is called by giving it as input 'CustomHopper-udr-v0' the environment is randomized each time it is initialized.

The train files are already ready to be ran, the test ones sometimes require arguments, usually --model to specify the desired model is the only required one; only the BAYRN test file takes as input the environment in which to test the model, allowing as values for --env udr, target and source (with source as default), the others have to be modified by hand to test on a specific environment, but usually point to the target one as it was the most used one for tests.

## Results comparison

<img src='https://github.com/AlirezaTalakoobi/sim2real_rl_robotics_mldl_22/blob/main/results_plots/modelcomp.png?raw=true' style="background-color: white;">