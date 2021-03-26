# Depth estimation from 4D light field videos
**Takahiro Kinoshita** and **Satoshi Ono**  
*Kagoshima University*  

International Workshop on Advanced Image Technology (IWAIT), Jan 2021  

[**[Paper]**](https://arxiv.org/abs/2012.03021)
[**[Code]**](https://github.com/mediaeng-lfv/LFV_Disparity_Estimation)
[**[Dataset]**](https://ieee-dataport.org/open-access/sintel-4d-light-field-videos-dataset)

---

## Abstract
Depth (disparity) estimation from 4D Light Field (LF) images
has been a research topic for the last couple of years.
Most studies have focused on depth estimation from static 4D LF images
while not considering temporal information, i.e., LF videos.
This paper proposes an end-to-end neural network architecture 
for depth estimation from 4D LF videos.
This study also constructs a medium-scale synthetic 4D LF video dataset that 
can be used for training deep learning-based methods.
Experimental results using synthetic and real-world 4D LF videos 
show that temporal information contributes to the improvement of 
depth estimation accuracy in noisy regions.

---

## Our Example Results
### Synthetic data
![synthetic_result](https://user-images.githubusercontent.com/37448236/107724284-df736a80-6d26-11eb-8891-563db2d6b960.gif)  
### Real-world data
![real_result](https://user-images.githubusercontent.com/37448236/107724517-5e68a300-6d27-11eb-9e32-dce1d3f08b71.gif)  

---

## Citation
If you find this useful for your research, please use the following.  
```bibtex
@inproceedings{kinoshita2021depth,
  title={Depth estimation from 4D light field videos},
  author={Kinoshita, Takahiro and Ono, Satoshi},
  booktitle={International Workshop on Advanced Imaging Technology (IWAIT) 2021},
  volume={11766},
  pages={117660A},
  year={2021},
  organization={International Society for Optics and Photonics}
}
```

*    *    *

## Our Dataset [[available at IEEE DataPort]](https://ieee-dataport.org/open-access/sintel-4d-light-field-video-dataset)  
![dataset](https://user-images.githubusercontent.com/37448236/107724659-bb645900-6d27-11eb-9b12-49377206892f.gif)  

In order to evaluate the performance of 4D LFVs depth estimation methods, 
we developed the Sintel 4D LFV dataset from the open-source movie *Sintel*.
It is difficult to accurately evaluate the effectiveness of 
deep learning-based 4D LFVs depth estimation methods 
in existing available 4D LFV datasets
due to small number of samples or no ground-truth disparity values is available.

The generated dataset consists of 23 synthetic 4D LFVs 
with 1,204x436 pixels, 9x9 views, and 20--50 frames, 
and has ground-truth disparity values in the central view, 
so that can be used for training deep learning-based methods.
Each scene was rendered with a *clean* pass 
after modifying the production file of *Sintel* 
with reference to [the MPI Sintel dataset](http://sintel.is.tue.mpg.de/).
A *clean* pass includes
complex illumination and reflectance properties including specular reflections,
such as smooth shading and specular reflections,
while bokeh, motion blur, and semi-transparent objects are excluded.

The 4D LFVs were captured by moving the camera
with a baseline of 0.01m
towards a common focus plane while keeping the optical axes parallel.
A ground-truth disparity value was obtained by transforming 
the depth value obtained in Blender.
The disparity values are in the range [0, 1] for most scenes, 
but up to 1.5 for some scenes.

### Scenes
[**GIF version page is here**](./gif_scenes_page). (**Note** that there are many large size GIFs.)

| Scene name    | Frames | Maximum disparity | RGB                                                | Disparity                                              |
|---------------|--------|-------------------|----------------------------------------------------|--------------------------------------------------------|
| ambushfight_1 | 20     | 0.366             | ![RGB](https://user-images.githubusercontent.com/37448236/107724787-08e0c600-6d28-11eb-8c79-baf8db9aba58.jpg) | ![Disparity](https://user-images.githubusercontent.com/37448236/107724791-0a11f300-6d28-11eb-8287-1e757060a5da.png) |
| ambushfight_2 | 21     | 1.522             | ![RGB](https://user-images.githubusercontent.com/37448236/107724886-36c60a80-6d28-11eb-8eab-76cc90948c3e.jpg) | ![Disparity](https://user-images.githubusercontent.com/37448236/107724889-375ea100-6d28-11eb-8557-608fc4cfdf94.png) |
| ambushfight_3 | 41     | 1.110             | ![RGB](https://user-images.githubusercontent.com/37448236/107725496-c28c6680-6d29-11eb-8277-9f17edf68fe3.jpg) | ![Disparity](https://user-images.githubusercontent.com/37448236/107725542-d6d06380-6d29-11eb-9406-00eaa2104f88.png) |
| ambushfight_4 | 30     | 1.005             | ![RGB](https://user-images.githubusercontent.com/37448236/107725633-01bab780-6d2a-11eb-9b1d-4d849e8b4485.jpg) | ![Disparity](https://user-images.githubusercontent.com/37448236/107725634-02ebe480-6d2a-11eb-949b-8799a3a232c7.png) |
| ambushfight_5 | 50     | 0.419             | ![RGB](https://user-images.githubusercontent.com/37448236/107755793-a35ffa00-6d66-11eb-9858-dea9988a67c8.jpg) | ![Disparity](https://user-images.githubusercontent.com/37448236/107755824-ac50cb80-6d66-11eb-8348-ac8bb464f721.png) |
| ambushfight_6 | 20     | 0.562             | ![RGB](https://user-images.githubusercontent.com/37448236/107756016-e9b55900-6d66-11eb-82fb-1f39976b55ad.jpg) | ![Disparity](https://user-images.githubusercontent.com/37448236/107756020-ea4def80-6d66-11eb-8d0c-ca321d4ef739.png) |
| bamboo_1      | 50     | 0.230             | ![RGB](https://user-images.githubusercontent.com/37448236/107756268-50d30d80-6d67-11eb-8fe7-882c50e8d260.jpg) | ![Disparity](https://user-images.githubusercontent.com/37448236/107756279-53cdfe00-6d67-11eb-8f92-48191c6b50c8.png) |
| bamboo_2      | 50     | 0.820             | ![RGB](https://user-images.githubusercontent.com/37448236/107756562-ba531c00-6d67-11eb-9217-1a43a6023bf2.jpg) | ![Disparity](https://user-images.githubusercontent.com/37448236/107756568-bd4e0c80-6d67-11eb-97de-f5297f9526a7.png) |
| bamboo_3      | 50     | 0.592             | ![RGB](https://user-images.githubusercontent.com/37448236/107757234-a0fe9f80-6d68-11eb-9cd3-f3e0612b499e.jpg) | ![Disparity](https://user-images.githubusercontent.com/37448236/107757229-9fcd7280-6d68-11eb-9be4-f60afd3d1b30.png) |
| chickenrun_1  | 50     | 1.005             | ![RGB](https://user-images.githubusercontent.com/37448236/107757491-f89d0b00-6d68-11eb-8b49-01be19449352.jpg) | ![Disparity](https://user-images.githubusercontent.com/37448236/107757499-fcc92880-6d68-11eb-8da0-67aca07f427d.png) |
| chickenrun_2  | 21     | 0.485             | ![RGB](https://user-images.githubusercontent.com/37448236/107757617-22563200-6d69-11eb-9275-dc37e2a52fe5.jpg) | ![Disparity](https://user-images.githubusercontent.com/37448236/107757619-23875f00-6d69-11eb-84aa-79d96720d798.png) |
| chickenrun_3  | 50     | 0.270             | ![RGB](https://user-images.githubusercontent.com/37448236/107757869-76611680-6d69-11eb-91b3-36b5521d5074.jpg) | ![Disparity](https://user-images.githubusercontent.com/37448236/107757881-78c37080-6d69-11eb-98d6-30537d3343dd.png) |
| foggyrocks_1  | 50     | 0.191             | ![RGB](https://user-images.githubusercontent.com/37448236/107758112-c5a74700-6d69-11eb-96ec-1c6d0f444355.jpg) | ![Disparity](https://user-images.githubusercontent.com/37448236/107758109-c50eb080-6d69-11eb-910e-33583a76b1a5.png) |
| foggyrocks_2  | 50     | 0.493             | ![RGB](https://user-images.githubusercontent.com/37448236/107758313-0b640f80-6d6a-11eb-9d3f-97231cd492a0.jpg) | ![Disparity](https://user-images.githubusercontent.com/37448236/107758311-0acb7900-6d6a-11eb-8764-490607079b82.png) |
| questbegins_1 | 40     | 0.882             | ![RGB](https://user-images.githubusercontent.com/37448236/107758475-45351600-6d6a-11eb-9d7a-d06f2f872f54.jpg) | ![Disparity](https://user-images.githubusercontent.com/37448236/107758474-449c7f80-6d6a-11eb-98db-df03423a17af.png) |
| shaman_1      | 50     | 2.148             | ![RGB](https://user-images.githubusercontent.com/37448236/107758665-8fb69280-6d6a-11eb-9e15-4a10d0b761c4.jpg) | ![Disparity](https://user-images.githubusercontent.com/37448236/107758673-92b18300-6d6a-11eb-80f1-54297764bd29.png) |
| shaman_2      | 50     | 1.191             | ![RGB](https://user-images.githubusercontent.com/37448236/107759072-22efc800-6d6b-11eb-8eba-39915c3613d7.jpg) | ![Disparity](https://user-images.githubusercontent.com/37448236/107759070-22573180-6d6b-11eb-96f9-f22c60de6a85.png) |
| shaman_3      | 50     | 0.954             | ![RGB](https://user-images.githubusercontent.com/37448236/107759227-56caed80-6d6b-11eb-976b-84fbb6171243.jpg) | ![Disparity](https://user-images.githubusercontent.com/37448236/107759233-592d4780-6d6b-11eb-95b0-43f69012a037.png) |
| shaman_b_1    | 48     | 1.348             | ![RGB](https://user-images.githubusercontent.com/37448236/107759377-9265b780-6d6b-11eb-9ea1-b30eef000397.jpg) | ![Disparity](https://user-images.githubusercontent.com/37448236/107759385-94c81180-6d6b-11eb-80b1-5c66c043aa78.png) |
| thebigfight_1 | 50     | 0.620             | ![RGB](https://user-images.githubusercontent.com/37448236/107759548-d1940880-6d6b-11eb-9812-37bb2199462c.jpg) | ![Disparity](https://user-images.githubusercontent.com/37448236/107759558-d3f66280-6d6b-11eb-9477-47bd682147ed.png) |
| thebigfight_2 | 50     | 0.302             | ![RGB](https://user-images.githubusercontent.com/37448236/107759704-0738f180-6d6c-11eb-9acb-e66f124b540e.jpg) | ![Disparity](https://user-images.githubusercontent.com/37448236/107759723-0a33e200-6d6c-11eb-8e33-462c31aab5d4.png) |
| thebigfight_3 | 50     | 0.317             | ![RGB](https://user-images.githubusercontent.com/37448236/107759855-3cddda80-6d6c-11eb-90c0-c1e08d9ac480.jpg) | ![Disparity](https://user-images.githubusercontent.com/37448236/107759865-3fd8cb00-6d6c-11eb-824e-46272c649dcf.png) |
| thebigfight_4 | 50     | 0.233             | ![RGB](https://user-images.githubusercontent.com/37448236/107759995-70b90000-6d6c-11eb-8ba1-682955bdf620.jpg) | ![Disparity](https://user-images.githubusercontent.com/37448236/107760005-731b5a00-6d6c-11eb-98e2-d152730a9948.png) |
