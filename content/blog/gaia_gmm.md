+++
author = "Joshua Wallace"
categories = ["research"]
tags = ["research, gaia"]
date = "2018-05-31"
description = "Applying Gaussian Mixture Models to Gaia Data"
linktitle = ""
title = "Gaussian Mixture Models and Cluster Membership"
type = "post"

+++

In my research, I look for planets around stars in a certain type of star cluster known as a [globular cluster](https://en.wikipedia.org/wiki/Globular_cluster). Globular clusters are the oldest known star clusters, nearly as old as the universe.  Although the locations of many of these clusters are well known and well measured, when observing there are still interloping stars in the foreground and background of these clusters.  Since we can't move these non-cluster stars out of the way, we're stuck observing them with the cluster stars.  The best we can do is identify in our images which stars belong to the cluster and which do not.

There are several methods that can be used to determine cluster membership.  For a variety of reasons, none of these methods work for all clusters, so the ideal membership classification method has to be determined cluster by cluster.  Fortunately for the cluster I am working on right now, there is one method that performs very well---the proper motion method---and there is a public data set that just got released that has the data necessary to use this method.

# The Proper Motion Method

First, let's understand what a "proper motion" is.  Stars appear fixed in their positions on the sky, but this is only an illusion.  All stars are in motion relative to Earth.  The stars are all so far away that this motion, except for a few stars, is impercetible to naked-eye observations on human time scales (centuries-millenia).  However, with precise instruments and enough time, the motion of stars across the sky can be measured.  The motion of a star on the plane of the sky is called its "proper motion."  (A second type of stellar motion, "radial velocity", is the motion of the star towards or away from Earth, perpendicular to the sky, but the technique to measure this motion is quite different from proper motion.)  The magnitude of measured proper motions is usually on the order of 1--10 milliarcseconds per year.  A milliarcsecond is 1 / 3,600,000 of a degree, just to get an idea of how incredibly small these motions on the sky typically are.

The proper motions of typical stars, as seen from the sun, are randomly oriented and usually quite small.  However, the stars in a star cluster move together (otherwise they wouldn't be clustered!) and so are not randomly oriented.  Additionally, some star clusters (including the one I am most interested in in my research) have higher than normal proper motions.  These two properties of cluster proper motions---local clustering and different-than-normal values---make it useful as a technique to distinguish which stars belong to a cluster and which are simply interlopers.


# The Data Set

The data I used for this analysis were obtained with the [Gaia satellite](http://sci.esa.int/gaia/), an incredible satellite with an ambitious mission to obtain proper motions for over a billion stars, the largest such dataset ever assembled by far.  The first data release from this mission was only a partial data release and didn't have proper motions for most of the stars I was interested in.  [The second data release](https://gea.esac.esa.int/archive/), available from April 2018, contained a much more complete set of proper motions and included the stars I was interested in.  The figure below shows the proper motions measured for stars in a section of sky in the direction of the cluster I'm studying.

{{< figure src="/img/2018/06/pm.png" title="Globular Cluster Proper Motion" 
    alt="Proper motions of stars in the direction of the globular cluster M4, showing the cluster of proper motions of the cluster stars. The cluster stars form a distinct grouping from the non-cluster stars." 
    caption="Proper motions for stars in the direction of the globular cluster M4.  RA stands for right ascension and dec stands for declination, which are perpendicular celestial coordinate axes;  mas stands for milliarcsecond.">}}

As we can see, there are two groupings of points in proper motion: one centered roughly on (-2, -5), and a second, more concentrated grouping centered roughtly on (-13, -20). It is that second, more concentrated grouping that is the stars that belong to the star cluster.  (This second grouping's proper motion is consistent with that previously measured for stars in the cluster, that's one of the reasons we know which of the two groupings belongs to the star cluster.) So, yay!  We have a robust data set with a very clear separation allowing for accurate classification.  How can we quantify this separation, though, to allow for automated classification?  We could separate the two groupings with a line and use that as the discriminator, but I will use another method---a Gaussian mixture model.


# Gaussian Mixture Model Method

Mixture models work with "mixed" datasets.  As an example of such a mixed dataset, let's consider the population of passengers at a train station. It consists of three sub-populations: those who came from outside the station to get on a train, those leaving the station after getting off a train, and those who are making a train after getting off another train.  These three sub-populations are mixed together, but with the right data, could be separated out (either probabilistically or deterministically depending on the data) despite the mixture.

Mixture models assume that there is such an underlying latent substructure to the dataset and then uses the data itself to discover this substructure.  This discovered substructure can then be used to determine membership of a data point in a given subpopulation.  Such membership determinations are not deterministic though, which is one of the advantages of mixture models; rather, for a data point, membership probabilities are determined for all the discovered subpopulations. In the train station example above, if we were able to interview everybody in the train station, then given they were all truthful we would have good enough data to determine exactly which of the three subpopulations everybody belonged to.  If, though, we weren't able to obtain such high-quality data and only had, say, a knowledge of their current location in the train station, then at best we could only determine probability a person had for belonging to each of the subpopulations.  If we wanted to run further analysis on our classification, it would be good to keep this uncertainty along for the ride, and the probabilistic output of mixture models allows us to do exactly that. For my star data, the probabilistic output is useful in deciding if and how to classify stars that don't fall near one of the two cluster cores.

The particular flavor of mixture model I chose to use is/was the Gaussian mixture model (GMM). This model assumes that the data arise from a mixture of Gaussian-distributed subpopulations.  A specified number of Gaussian distributions are then fit to the data (typically using the [expectation-maximization algorithm](https://en.wikipedia.org/wiki/Expectation%E2%80%93maximization_algorithm) to maximize the likelihood).  The parameters that are fit are the distribution means and variances. 

Why did I choose a GMM instead of another type of mixture model?  First, looking at the data when plotted as above, the clustering of the data seems at least Gaussian-ish: two dense, central cores with point density tapering off from the cores with rougly elliptical contour lines. Second, the proper motions of the objects belonging to the star cluster should be nearly exactly the same, so when factoring in measurement errors, we'd expect the distribution of the measured proper motions to be Gaussian anyway.  The proper motions of the stars not belonging to the star cluster posses both intrinsic randomness and measurement errors, so these points should be at least roughly Gaussian distributed as well. Thus, a GMM should be able to model the data well.

(A brief aside: the larger scatter in the x-direction compared to the y-direction is real from the measurements and is a consequence of the coordinate system used being spherical, RA being the azimuthal angle, and stars being about 26 degrees away from the coordinate system's equator.)

# The Results

Let's see how well the GMM works.  I used the `scikit-learn` package's `mixture.GaussianMixture` for this.

{{< figure src="/img/2018/06/gmm.png" title="Cluster Membership Probabilities Using A GMM" 
    alt="The Gaussian mixture model is able to cleanly separate stars that are members of the globular cluster from those that aren't, while assigning membership probabilities to each star to allow us to identify which stars' membership is ambiguous." 
    caption="Proper motions for stars in the direction of the globular cluster M4, with cluster membership probabilities calculated by fitting a GMM.  Green denotes high probability of cluster membership while blue denotes low probability of membership and red denotes ambiguous membership. Note that this is zoomed in relative to the previous figure.">}}

---

I think the GMM did very well, especially considering that there were no hyperparameters to optimize and so implementation was rather quick.  The model has discovered and separated the two clusters. One important note is that mixture models identify clusters in the data but do not attach any sort of "labels" to the cluster; labels have to be assigned after the fact.  Relatedly, different initializations for the the model fitting can lead to the same clusters being identified but having different labels.  As an example of what I mean, in the present case, that would be like different model initializations leading to the blue and the green colors being reversed.  Thus, not only do labels need to be assigned after performing the model fit, but those labels are only guaranteed to be good for that particular fit.  If the model is refitted with a different initialization, what the model identifies only as "cluster 1" and "cluster 2" may be reversed and thus the labels would need to switch which model cluster they belong to. 

One thing I notice is that the red points, with equal probability of belonging to either cluster, seem more likely to belong to the green point cluster based on the density of points in that vicinity.  A possible cause of this is one or both of the clusters not actually being Gaussian due to, e.g., unaccounted systematics in the measurements.  I haven't taken a closer look to figure out what the cause may be since the clustering has worked well enough. It's possible to get around this by choosing a thresholding other than 50% (perhaps 40%) to count something as belonging to the star cluster.

# Conclusion

Using a GMM for cluster membership determination has already been extremely useful in my own research.  The power of this incredible dataset from the Gaia mission, coupled with a basic clustering algorithm, has allowed a relatively small amount of effort to produce a cluster membership catalog that is better than anything previously published.  That is the power of a dataset that is orders of magnitude better than those available previously.