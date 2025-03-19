library(dplyr)

clinical <- read.csv('survival_TCGA-LIHC_XENA.csv')
RNA <- readRDS('RNA-LIHC.rds')
genes <- c("AARS2", "ARHGEF11", "RABEPK", "ATP6V0A2") # top4 genes

colnames(clinical)
rownames(clinical) <- clinical$patientID

# RNA 
colnames(RNA) <- substr(colnames(RNA), 1, 12)
expr <- RNA[genes,]
head(expr)
write.csv(expr,'RNA_4genes.csv')

#filter information
patient_barcode <- intersect(colnames(expr),rownames(clinical))
meta_clinical <- clinical[patient_barcode, c('OS.time_5y','OS_5y',
                                             'OS.time','OS',
                                             "neoplasm_histologic_grade",
                                             "gender.demographic",
                                             "age_at_index.demographic")]
meta_clinical <- cbind(meta_clinical, as.data.frame(t(expr[,patient_barcode])))
dim(meta_clinical) #365,11


# single gene Cox-------------------------------------------------------------------------
library(ggplot2)
library(survival)
library(survminer)

genes <- c("AARS2", "ARHGEF11", "RABEPK", "ATP6V0A2")
meta_stage <- meta_clinical

for (gene in genes){
  meta_stage$gene <- meta_stage[,gene]
  coxfit <- coxph(Surv(OS.time_5y, OS_5y)~gene, data=meta_stage)
  print(gene)
  print(summary(coxfit))
}

# combination 4 genes -----------------------------------------------------

meta_stage <- meta_clinical %>% 
  mutate(risk_score = 1.07818*AARS2 + ARHGEF11*1.04578 + 0.998186*RABEPK + 1.6646*ATP6V0A2)

(threshould <- median(meta_stage$risk_score)) # median
meta_stage$risk_score <- ifelse(meta_stage$risk_score > threshould,1,0) #group

sfit <- survfit(Surv(OS.time_5y, OS_5y)~risk_score, data=meta_stage)

p <- ggsurvplot(sfit,
                conf.int=F, pval=TRUE, risk.table = TRUE,
                title = gene,
                font.main=18, font.x=16, font.y=16,
                palette = c("#fca311", "#1a759f"),
                legend.labs = c("risk_score=0", "risk_score=1"),
                legend.title = "",
                legend = c(0.8,0.75)
)
ggsave(plot = p$plot, filename = paste0("singleGene/OS_median_risk.pdf"))


# HE features ---------------------------------------------------------------------

meta_stage <- meta_clinical2 %>% mutate(risk_score = pred_HCC_rate)

(threshould <- median(meta_stage$risk_score)) # median
meta_stage$risk_score <- ifelse(meta_stage$risk_score > threshould,1,0) #group

sfit <- survfit(Surv(OS.time_5y, OS_5y)~risk_score, data=meta_stage)

p <- ggsurvplot(sfit,
                conf.int=F, pval=TRUE, risk.table = TRUE,
                title = 'HE features',
                font.main=18, font.x=16, font.y=16,
                palette = c("#fca311", "#1a759f"),
                legend.labs = c("risk_score=0", "risk_score=1"),
                legend.title = "",
                legend = c(0.8,0.75)
)
ggsave(plot = p$plot, filename = "HE+gene/OS_median_HE.pdf")

# combination 4 genes and HE ----------------------------------------------

meta_stage <- meta_clinical2 %>% 
  mutate(risk_score = pred_HCC_rate + 1.07818*AARS2 + ARHGEF11*1.04578 + 0.998186*RABEPK + 1.6646*ATP6V0A2)

(threshould <- median(meta_stage$risk_score))  # median
meta_stage$risk_score <- ifelse(meta_stage$risk_score > threshould,1,0) # group

sfit <- survfit(Surv(OS.time_5y, OS_5y)~risk_score, data=meta_stage)

p <- ggsurvplot(sfit,
                conf.int=F, pval=TRUE, risk.table = TRUE,
                title = 'HE +4 genes',
                font.main=18, font.x=16, font.y=16,
                palette = c("#fca311", "#1a759f"),
                legend.labs = c("risk_score=0", "risk_score=1"),
                legend.title = "",
                legend = c(0.8,0.75)
)
ggsave(plot = p$plot, filename = paste0("HE+gene/OS_median_risk.pdf"))
