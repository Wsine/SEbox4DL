<img align="right" height="200" src="https://s1.52poke.wiki/wiki/thumb/3/3b/077Ponyta.png/300px-077Ponyta.png">

# SEbox4DL

Project Code: Ponyta

For the technical details, please refer to the following publication.

## Publication

**Plain Text:**

Z. Wei, H. Wang, Z. Yang and W. K. Chan, "SEbox4DL: A Modular Software Engineering Toolbox for Deep Learning Models," 2022 IEEE/ACM 44th International Conference on Software Engineering: Companion Proceedings (ICSE-Companion), 2022, pp. 193-196, doi: 10.1109/ICSE-Companion55297.2022.9793795.

**BibTex:**

```bibtex
@INPROCEEDINGS{9793795,
  author={Wei, Zhengyuan and Wang, Haipeng and Yang, Zhen and Chan, W.K.},
  booktitle={2022 IEEE/ACM 44th International Conference on Software Engineering: Companion Proceedings (ICSE-Companion)}, 
  title={SEbox4DL: A Modular Software Engineering Toolbox for Deep Learning Models}, 
  year={2022},
  volume={},
  number={},
  pages={193-196},
  doi={10.1109/ICSE-Companion55297.2022.9793795}
}
```

## How to take a try

Click this button below to open this project in google cloud shell

[![Open this project in Cloud
Shell](http://gstatic.com/cloudssh/images/open-btn.png)](https://console.cloud.google.com/cloudshell/open?git_repo=https://github.com/Wsine/SEbox4DL&page=editor&open_in_editor=README.md)

Install the dependencies

```bash
python3 -m virtualenv venv && source venv/bin/activate && pip3 install -r requirements.txt
export PYTHONPATH=$(pwd)
streamlit run app/main.py --server.headless true --server.port 8080
```

When it starts, click the PREVIEW button on the top-right corner and preview via the port `8080`

## Learn how to use

a YouTube video is shown for the demonstration. just click the image below.

[![SEbox4DL](https://user-images.githubusercontent.com/8842278/181018903-56be1deb-8b5b-4845-b3f9-19bd5f611ea3.png)](https://www.youtube.com/watch?v=EYeFFi4lswc)

## Acknowledge

- [Streamlit](https://streamlit.io/)
- [Pytorch Hub](https://pytorch.org/hub/)
- [TorchVision](https://pytorch.org/vision/stable/index.html)

