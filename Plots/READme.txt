Created 3 plots of 5 different distribution (named accordingly).
Each image file named dist1,2,3 has different number of nodes between 10-20.

I have used tsne with 2 components and used seaborn scatterplot to display the compressed features.
Used tsne with exact method and varied perplexity for different number of nodes (for best visualization possible).
Number of features I have used is 50 as tSNE best performs with these order of features.
I have plotted total of 6 plots for each training loop, everyone of which is at different point of time.
It seems like for most of the distributions there is no major reduction in training loss which implies that there is no clear visualization of the diffrences between plots.
But there is subtle convergence of points of the same class, which can be observed in most cases.



