# AI-model-for-eHCC
## Background
Detection of early hepatocellular carcinoma (eHCC) is important for timely treatment and improved prognosis. However, it is challenging to distinguish eHCC from pre-malignant high-grade dysplastic nodules (HGDN). Here we developed an artificial intelligence (AI) derived computational framework to identify potential biomarkers and built classification models for eHCC and HGDN. A two-stage multiscale deep learning model (TMC-net) captured the subtle features based on H&E images, and outperformed the pathology foundation model and traditional histopathological features. The learned features were consistent with clinical diagnostic criteria and could be highlighted on the virtual images, assisting junior pathologists in improving the diagnostic accuracy. Four marker genes were screened through comparative transcriptome analysis. The multimodal model based on marker genes and histopathological features achieved AUROC of 0.8875 and 0.9500 on the internal and external test sets, respectively. We confirmed the morpho-phenotype correlations of these genes and found that the multimodal features were associated with patient prognosis in a broader HCC cohort.  This study reveals histopathological and transcriptomic features of eHCC, and provides an optimized AI solution for assistant diagnosis.

## Usage
The modeling and analysis of HE image were developed in Python (3.9). The pre-processing of RNA data, differential gene expression and functional analysis were developed in R version (4.1.3).

For detailed information on the specific modeling and statistical analysis methods, please refer to the supplementary methods.

The **deep learning model TMC-net** modeling code can be obtained from /AI-model-for-eHCC/TMC-net.   
**Transcriptomics**: Codes for differential gene expression, functional analysis, and candidate biomarker screening are available at /AI-model-for-eHCC/RNA.  
**Multimodal model** code: Multimodal modeling, comparison with the unimodal model, and modeling for different numbers of genes can be obtained from /AI-model-for-eHCC/multimodal_model.  
The codes for **survival analysis** and **morphological correlations** are available at /AI-model-for-eHCC/supplementary.  

## Citation
If you use code or data in this project, please cite:

- Jing SY, LI XL, et al. 2025. "Multimodal AI model for early detection of hepatocellular carcinoma". 

## Maintainers
[@jessiya](https://github.com/jessiya825), 
[@LittlePlumLi](https://github.com/LittlePlumLi), 
