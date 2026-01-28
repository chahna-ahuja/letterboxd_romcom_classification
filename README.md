# Boy Meets Girl Meets Natural Language Processing: Exploring Binary Text Classification of Rom-Com Subgenres Using Movie Synopses in Letterboxd Dataset
# Author: Chahna Ahuja 
# Final Class Project for Computational Linguistics (2025-26) at KU Leuven (Msc Digital Humanities)

## Project Overview and Scope

One of the foundational scholarly projects of the digital humanities has been to leverage the possibilities offered by computing technologies to develop critical tools, infrastructures, and methodological pipelines that enable new forms of inquiry within the humanities. As [Piotrowski (2012)](#https://link.springer.com/chapter/10.1007/978-3-031-02146-6_2)  argues, the increasing convergence between natural language processing (NLP) and digital humanities reflects a shift on the NLP side toward genres beyond newspaper and newswire texts towards biomedical corpora, forum posts, and social media data, where computational linguists have increasingly turned their attention to historical texts and other textual forms of interest to the humanities and social sciences [(Piotrowski 2012, 7)](#(#https://link.springer.com/chapter/10.1007/978-3-031-02146-6_2).

This project engages with one such textual data, that is, the metadata produced by the social film platform Letterboxd. Originally conceived as a “Goodreads for film,” Letterboxd is as a social cataloguing platform  for movies, where cinephiles maintain personal film diaries, write and engage with movie reviews, and curate recommendation lists. Its large-scale collection of movie metadata, including synopses, tags, and user-generated reviews, constitutes a rich corpus for computational text analysis. From a digital humanities perspective, Letterboxd offers a valuable site for examining film genres, audience reception, fan discourses, and curatorial practices such as review-writing in contemporary digital culture.

**The scope of this project is to perform binary text classification of movie genres based on plot synopses of films released in the twenty-first century (2000–2025)**. Text classification can be defined as class of supervised machine-learning methods that assign predefined categories (two in the case of binary categories) to textual data using computational classifiers. Within both computational linguistics and digital humanities, text classification is a well-established methodological approach. Recent work by [Rich (2025)](#https://www.degruyterbrill.com/document/doi/10.1515/opth-2025-0052/html), for example, demonstrates the applicability of neural-network-based text classification in religious and medieval studies. Earlier digital humanities work on genre classification dates back projects such as Blackstock and Spitz (2008), who classified movie scripts by genre using statistical NLP approaches such as Naive Bayes classifiers.

While genre classification has been extensively explored in NLP, most existing studies focus on multi-label genre prediction or broad genre taxonomies, often using film scripts or large plot datasets. A smaller body of work has examined movie genre classification using plot summaries specifically (e.g., [Blackstock and Spitz 2008](#https://www.semanticscholar.org/paper/Classifying-Movie-Scripts-by-Genre-with-a-MEMM-Blackstock-Spitz/35355f27b62269bf8ffb260bbb5ca6a74d4ef1b9); [Abimbola 2020](#https://github.com/Wonuabimbola/movie-genre-prediction?tab=readme-ov-file); [Kumar et al. 2022](#https://link.springer.com/article/10.1007/s11042-022-13211-5)), yet these studies tend to view genre as a predictability classifier, rarely engaging with genre as a culturally constructed category that also includes subgenres like romantic comedy, horror-comedies,documentary-drama, etc.

This project focuses on romantic comedy for binary text classification using statistical and neural paradigms in machine learning. Despite its highly commercialized and conventional narrative structures and its canonical cultural significance in popular cinema, this subgenre has, to the best of my knowledge, not received focused attention as a standalone classification task. This project isolates the romantic-comedy (rom-com) genre and frames genre detection as a binary classification task—distinguishing rom-coms from non-rom-com films using synopses of movies released from 2000 to 2025 in the Letterboxd dataset.

---
## Project Methodologies and Models: Naive Bayes and DistilBERT

To explore the binary text classification of romantic comedies versus non-romantic comedies, this project employs two contrasting NLP approaches: **(1) a Naive Bayes classifier using both bag-of-words and TF-IDF, and (2) DistilBERT, a transformer-based language model**.

Naive Bayes represents a classical probabilistic approach to text classification and has been widely used in foundational NLP research, including early work on movie genre classification (Blackstock and Spitz 2008). Its suitability for this project lies in both its methodological simplicity and interpretability.

Since Letterboxd synopses function as marketable metadata texts, designed to attract audiences by signalling genre conventions, Naive Bayes allows for close examination of surface-level lexical patterns using **Bag-Of-Words (BOW) and Term Frequency-Inverse Document Frequency (TF-IDF) approaches**. In the former, the model solely focuses on word frequency, while the latter can be considered an enhancement of the BOW model as it also considers the rarity of words across the dataset by not only considering word frequency but also using inverse document frequency to adjust the weight of each word before transforming it into a vector. Together, these approaches make it possible to identify which lexical features most strongly signal the rom-com genre. From a digital humanities perspective, this interpretability is particularly valuable, as it supports critical reflection on how genre is linguistically encoded within movie synopsis metadata.

By contrast, DistilBERT is a contemporary neural, transformer-based language model that captures contextual and semantic relationships between words beyond frequency-based representations. Its inclusion in the project allows for an assessment of whether deeper contextual embeddings offer advantages when classifying genre from synopsis texts.

---

## Methodological Rationale for DH

Rather than assuming the inherent superiority of neural models, the comparative analysis between Naive Bayes and DistilBERT foregrounds key methodological trade-offs between interpretability and representational depth. This comparative analysis aligns with digital humanities commitments to methodological reflexivity, positioning computational models not only as tools for predictive accuracy but as critical instruments for interrogating how cultural categories such as genre are encoded, stabilized, and made legible in marketable metadata texts like movie synopsis.

--- 
## Dataset Information and Reproducibility of Project 

The downloaded dataset can be found in [my other Github repository](#https://github.com/chahna-ahuja/letterboxd_project/blob/main/data/letterboxd_all_movies.parquet) as this project uses the same Letterboxd dataset. Please run the script in this repository with a stronger GPU or Google Colab to reproduce the analysis of this project. 
