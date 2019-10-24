#Structure-Preserving Color Normalization and Sparse
Stain Separation for Histological Images (Vahadane normalization)



### Abhishek Vahadane*, Tingying Peng*, Amit Sethi, Shadi Albarqouni, Lichao Wang, Maximilian Baust,
###Katja Steiger, Anna Melissa Schlitter, Irene Esposito, and Nassir Navab


code: python3

code name: vahadane.py

Abstract—Staining and scanning of tissue samples for microscopic examination is fraught with undesirable color variations
arising from differences in raw materials and manufacturing
techniques of stain vendors, staining protocols of labs, and color
responses of digital scanners. When comparing tissue samples,
color normalization and stain separation of the tissue images can
be helpful for both pathologists and software. Techniques that
are used for natural images fail to utilize structural properties of
stained tissue samples and produce undesirable color distortions.
The stain concentration cannot be negative. Tissue samples are
stained with only a few stains and most tissue regions are characterized by at most one effective stain. We model these physical
phenomena that define the tissue structure by first decomposing
images in an unsupervised manner into stain density maps that
are sparse and non-negative. For a given image, we combine its
stain density maps with stain color basis of a pathologist-preferred
target image, thus altering only its color while preserving its
structure described by the maps. Stain density correlation with
ground truth and preference by pathologists were higher for
images normalized using our method when compared to other
alternatives. We also propose a computationally faster extension
of this technique for large whole-slide images that selects an
appropriate patch sample instead of using the entire image to
compute the stain color basis.


