library(dplyr)

# load data
expr <- read.table('rawData/genes.count_table.withName', header=TRUE)
dim(expr)
sampleInfo <- read.csv('rawData/sampleInfor.csv',header=TRUE)
dim(sampleInfo)

# Use aggregate to merge genes with the same gene in the symbol column
expr <- aggregate(expr, by=list(geneName), FUN=max)  
rownames(expr) <- expr$`Group.1`
expr <- expr[-1]


# remove batch ------------------------------------------------------------

library(sva)

expr <- readRDS('expr_filter.rds')
sampleInfo <- readRDS('sampleInfo_filter.rds')

expr_combat <- ComBat_seq(as.matrix(expr),
                          batch = sampleInfo$RNA.Seq_Time2,
                          group = sampleInfo$group)

# PCA ---------------------------------------------------------------------

library(FactoMineR)
library(factoextra)
library(repr)

options(repr.plot.width=6,repr.plot.height=5)
expr2 <- scale(t(expr))  # 统一量纲标准化
expr2 <- expr2[, colSums(is.na(expr2)) == 0] # 删除NA列

pre.pca <- PCA(expr3, graph = FALSE)
fviz_pca_ind(pre.pca,
             geom= "point",
             col.ind = sampleInfo$RNA.Seq_Time,
             addEllipses = TRUE,
             legend.title="Group",
             title = 'all patients')


# DEGs-------------------------------------------------------------------------
library(DESeq2) 
library(dplyr)
library(stringr)


# load data
expr <- readRDS('expr_filter.rds')
sampleInfo <- readRDS('sampleInfo_filter.rds')

sampleInfo <- sampleInfo %>% filter(group %in% c('DN','HCC'))
expr <- expr[,rownames(sampleInfo)]

dim(expr)
dim(sampleInfo)

sampleInfo$RNA.Seq_Time2 <- as.factor(sampleInfo$RNA.Seq_Time2)

i <- 'HCC'
sampleInfo$design <- NA
sampleInfo[which(sampleInfo$group == i),'design'] <- 'trt'
sampleInfo[is.na(sampleInfo$design),'design'] <- 'ctl'
sampleInfo$design <- as.factor(sampleInfo$design)
# dds obj
dds <- DESeqDataSetFromMatrix(countData = round(expr), 
                              colData = sampleInfo, 
                              design= ~ RNA.Seq_Time2 + design) 

dds <- DESeq(dds)  #标准化
res <- results(dds,
               contrast = c('design', "trt", "ctl"),
               pAdjustMethod = "fdr")
res <- res[order(res$padj),]  #取P值小于0.05的结果
diff_gene_deseq2 <- subset(res, padj<0.05)

resdata <- merge(as.data.frame(diff_gene_deseq2),
                 as.data.frame(counts(dds,normalize=TRUE)),
                 by="row.names", sort=FALSE)

saveRDS(resdata, file='DEG_HCCVDN_DEseq2.rds')

# pathway-------------------------------------------------------------------------

library(ggplot2)
library(dplyr)
library(msigdbr)
library(clusterProfiler) # env: R4.3

load('rawData/geneSets_DataBase.RData') # gsKEGG
DEG <- readRDS('DEG_DNVHCC_DEseq2.rds')
head(DEG)

i <- 1

types <- c('Y','DN','KEGG')
data_sort <- DEG %>% 
  filter(group==types[i], padj<0.05) %>% 
  arrange(desc(log2FoldChange))  

H_df <- msigdbr(species = "Homo sapiens",category = "H")[c('gs_name', 'gene_symbol')]
head(H_df, 2)

Hallmark <- enricher(data_sort$Row.names,
                     TERM2GENE = H_df,
                     pvalueCutoff = 0.5) 

kegg <- enricher(data_sort$Row.names,
                 TERM2GENE = kegg_df,
                 pvalueCutoff = 0.5) 

p <- dotplot(kegg, showCategory = 10)
p <- p+ scale_y_discrete(labels = function(x) str_to_title(x))


# expression test -------------------------------------------------------------------------

library(dplyr)
library(ggplot2)
library(ggsignif)

expr_combat <- readRDS('count_combat.rds')
sampleInfo <- readRDS('sampleInfo_filter.rds')
myGene <- c('AFP','GPC3','HSPA1A','HSPA2','HSPE1') 

myGene <- intersect(rownames(expr_combat),myGene)

expr <- expr_combat[myGene,]
dim(expr)
expr <- as.data.frame(t(expr))
expr$label <- sampleInfo$group


for (i in 1:5){
  gene <- myGene[i]
  
  expr[,"gene"] <- expr[i]
  plotdf <- expr %>% filter(label %in% c('DN','HCC')) %>% tidyr::drop_na(gene) 
  p <- ggplot(plotdf, aes(x=label, y=gene, fill=label))+
    geom_violin()+
    theme_classic(base_line_size = 1)+
    labs(title=gene, x="", y="Expression level")+
    theme(plot.title = element_text(size=20,face = "italic",),axis.text = element_text(size=15),
          legend.position="none")+
    geom_signif(comparisons = list(c("DN", "HCC")),
                map_signif_level = TRUE)
  assign(paste0("p",i),p)
  ggsave(paste0("IHC/Expression_",gene,".pdf"), p, height = 5,width = 5 )
  
}
