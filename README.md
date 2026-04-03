
# CoMuCo
Official repository for the AAAI 2026 paper "Cross-Domain Few-Shot Learning via Multi-View Collaborative Optimization with Vision-Language Models"

## 🎉 The Code is Here! 

Remember when we said it was coming soon? **The wait is over!** 🥳 
Thank you so much for your patience while we got things pretty and organized. The code is now officially live! 

*(And just a reminder, our benchmark dataset is also public!)*
**📊 Benchmark Dataset Link:**
https://huggingface.co/datasets/Kxon/CoMuCo_cross_domain_benchmark

---

## 🚀 Getting Started

Our implementation is built upon the elegant **CoOp** framework. To run CoMuCo, you simply need to merge our code into the CoOp repository. 

Here is a quick step-by-step guide:

### 1. Setup the Environment
First, you'll need to set up the CoOp environment and its prerequisite, the `Dassl` library. Please follow their official installation instructions:
* Clone [CoOp](https://github.com/KaiyangZhou/CoOp)
* Install [Dassl.pytorch](https://github.com/KaiyangZhou/Dassl.pytorch)

### 2. Merge CoMuCo into CoOp
Once CoOp is set up, download our repository and move our files into the CoOp directory:
* Copy the contents of our `trainers/` folder into CoOp's `trainers/` directory.
* Copy the contents of our `configs/` folder into CoOp's `configs/` directory.

### 3. Register the Trainer
To let the framework recognize our model, open CoOp's `train.py` and import our trainer at the top of the file along with the other imports.

### 4. Start Training!
That's it! You can now train and evaluate CoMuCo exactly following the standard CoOp workflow.

---

## 🙏 Acknowledgements

Our code is heavily inspired by and built upon several fantastic open-source projects. We would like to express our sincere gratitude to the authors of the following repositories for their invaluable contributions to the community:

* **[CoOp (Context Optimization)](https://github.com/KaiyangZhou/CoOp)** 
* **[Dassl.pytorch](https://github.com/KaiyangZhou/Dassl.pytorch)**
* **[Tip-Adapter](https://github.com/gaopengcuhk/Tip-Adapter)**

---

## 📝 Citation

If you find our code, benchmark, or paper useful in your research, please consider citing our work:

```bibtex
@article{chen2025cross,
  title={Cross-Domain Few-Shot Learning via Multi-View Collaborative Optimization with Vision-Language Models},
  author={Chen, Dexia and Zhang, Wentao and Zhu, Qianjie and Hu, Ping and Li, Weibing and Zhang, Tong and Wang, Ruixuan},
  journal={arXiv preprint arXiv:2508.12861},
  year={2025}
}
```

