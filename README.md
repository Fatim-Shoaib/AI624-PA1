# AI-624 PA1

# Task 0
- Use pre-trained VGG16-BN Model, pre-trained on CIFAR10 and CIFAR100
- Download and load the CIFAR Datasets and apply appropriate transformations
- Create training and test loaders
- Choose appropriate batch size
- Use the loaded VGG16-BN model and CIFAR dataset to perform the following:
  1.  Peak GPU memory, average GPU memory (averaged across operations), end-to-end inference latency (ms) pre batch using torch.profiler
  2.  Energy useage (both CPU and GPU) using pyJoules
  3. Compute serialized model size in MB
  4. Record number of MACs using torchprofile
  Average the results across a few batches for (1) and (2). Choose reasonable batch size for (c)
- Evaluate Top-1 and Top-5 accuracy on test sets and train sets. Confirm these are close to the ones reported in the original source.


# Task 1

# Resources

## Task 0
- py tutorials: https://docs.pytorch.org/tutorials/
- profiling tutorial: https://docs.pytorch.org/tutorials/intermediate/fx_profiling_tutorial.html
- profiling tutorial: https://docs.pytorch.org/tutorials/beginner/hta_trace_diff_tutorial.html
- profiling tutorial: https://docs.pytorch.org/tutorials/beginner/hta_intro_tutorial.html
- profiling tutorial: https://docs.pytorch.org/tutorials/beginner/profiler.html

