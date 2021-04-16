# load-prediction-dsp

This repository is accompanying a paper submitted to the IC2E 2021 conference.

### Title: Evaluation of Load Prediction Techniques for Distributed Stream Processing

### Abstract:

Distributed Stream Processing (DSP) systems enable processing large streams of continuous data to produce results in near to real time. They are an essential part of of many data-intensive applications and analytics platforms.
The rate at which events arrive at DSP systems can vary considerably over time, which may be due to trends, cyclic, and seasonal patterns within the data streams. A priori knowledge of incoming workloads enables proactive approaches to resource management and optimization tasks such as dynamic scaling, live migration of resources, and the tuning of configuration parameters during run-times, thus leading to a potentially better Quality of Service.

In this paper we conduct a comprehensive evaluation of different load prediction techniques for DSP jobs. 
We identify three use-cases and formulate requirements for making load predictions specific to DSP jobs. Automatically optimized classical and Deep Learning methods are being evaluated on nine different datasets from typical DSP domains, i.e. the IoT, Web 2.0, and cluster monitoring. 
We compare model performance with respect to overall accuracy and training duration.
Our results show that the Deep Learning methods provide the most accurate load predictions for the majority of the evaluated datasets.

---

The data used for the experiments is in the data sub folder.

---

The experiments are separated in Deep Learning and classic Time Series Forecasting techniques, including Facebooks Prophet.

The respective sub folder contains a conda environment.yaml to set up the experimental environment. Additionally, the sub folders contain their own README files explaining the process to run the experiments.

---

Results achieved for our paper submission are included in the respective sub folders.

---

Reach out to <kordian.gontarska@hpi.de> if you encounter problems or have questions.