# LM-Alloy-Superconductor
<center>
Chen Hua<sup>a,b</sup>, Jing Liu<sup>a,b,c</sup>*
</center>    

<sup>a)</sup>Key Lab of Cryogenic Science and Technology, Technical Institute of Physics and Chemistry, Chinese Academy of Sciences, Beijing 100190, China.    
<sup>b)</sup>School of Future Technology, University of Chinese Academy of Sciences, Beijing 100049, China.    
<sup>c)</sup>School of Biomedical Engineering, Tsinghua University, Beijing 100084, China.    
*Correspondence: jliu@mail.ipc.ac.cn    
Author email: h_uachen@163.com


# Introduction

# Condition
The data and the codes can be used under the condition that you cite the following paper. Also see Licence.
```
%\cite{journal.volume.pages}
@article{journal.volume.pages,
  title = {},
  author = {},
  journal = {},
  volume = {},
  issue = {},
  pages = {},
  numpages = {},
  year = {},
  month = {},
  publisher = {y},
  doi = {},
  url = {}
}
```

# Data for Model
Data for Model is stored in ```LM-Alloy-Superconductor/Data```.    

(1) The dataset ```20240322_MDR_OAndM.csv```  is obtained from:  
  ```
  1. https://doi.org/10.48505/nims.3739
  2. Science and Technology of Advanced Materials 2015, 16 (3), 033503.
  3. Phys. Rev. B 2021, 103 (1), 014509.
```  
(2) The dataset  ```mdr.csv``` is obtained in ```20240322_MDR_OAndM.csv``` whose Type contain "Availabel".    
(3) The dataset  ```mdr_clean.csv``` is preprocessed from mdr.csv.    
(4) The file ```alloy_element.csv``` contains the symbols of the elements that have appeared in ```mdr_clean.csv```, which are divided according to metal and non-metal.    
(5) The file ```数据清洗.txt``` contains records of data preprocessing.    
(6) The file ```mdr_duplicated.csv``` contains part of formulas in ```mdr_clean.csv``` with mutiple T<sub>c</sub>.


# Code for Model
The code can be provided upon request from the author(s).
