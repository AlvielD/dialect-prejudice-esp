# Dialect Prejudice in Language Models

This repository is a forked version of the one done by Valentino Hofmann for the paper [Dialect prejudice predicts AI decisions about people's character, employability, and criminality](https://arxiv.org/abs/2403.00742). You can find the original version of the repository [here](https://github.com/valentinhofmann/dialect-prejudice). This is just an academic version I performed for my project in _Exploring Toxic and Hate Bias in Large Language Models (LLMs)_ for the University of Bologna (UNIBO).

## Summary

The relevant files for this project (the ones that I modified) are those related to the Matched Guise Probing method. These files are the following ones:
- Demo [notebook](https://github.com/AlvielD/dialect-prejudice-esp/blob/main/demo/matched_guise_probing_demo.ipynb) where the experiment is performed.
- The file implementing all the methods for the computation of probabilities and other important functions. [helpers.py](https://github.com/AlvielD/dialect-prejudice-esp/blob/main/probing/helpers.py)
- The adjectives from the Princeton Trilogy translated into spanish with their respective forms. [katz_esp.json](https://github.com/AlvielD/dialect-prejudice-esp/blob/main/data/attributes/katz_esp.json)
- The pairs in Latino American Spanish and Peninsular Spanish. [pesp_lesp.txt](https://github.com/AlvielD/dialect-prejudice-esp/blob/main/data/pairs/pesp_lesp.txt)
- The prompts translated in spanish. [prompting.py](https://github.com/AlvielD/dialect-prejudice-esp/blob/main/probing/prompting.py)

Read the notebook for more information on the performed experiment.

The full report for the project can be found [here](https://github.com/AlvielD/dialect-prejudice-esp/blob/main/Ethics_Project_Report.pdf)