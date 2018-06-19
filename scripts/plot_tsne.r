library("ggplot2")
library("reshape2")
library("latex2exp")
library("randomcoloR")


# Plot the t-sne plot for one dataframe
plot_plate = function(df, channel, x_low, x_high, y_low, y_high,
                      fold_size=30){
    # Make sure DMSO is always in the sub-plots
    x = split(df, df$label)
    dmso_df = x[[sprintf("%s_DMSO", channel)]]
    x[sprintf("%s_DMSO", channel)] = NULL
    
    for (i in 1:ceiling(length(x) / fold_size)){
        print(sprintf("Plotting %d of %s", i, channel))
        upper = min(i * fold_size, length(x))
        cur_df = rbind(dmso_df, x[[(i - 1) * fold_size + 1]])
        for (t in ((i - 1) * fold_size + 2) : upper){
            cur_df = rbind(cur_df, x[[t]])
        }
        
        # Plot the current dataframe
        cur_df$label <- gsub(sprintf("%s_DMSO", channel),
                             '00_DMSO', cur_df$label)
        
        # Generate the most distinctive colors
        my_color = distinctColorPalette(length(unique(cur_df$label)))
    
        # Make sure DMSO has the same color across all the plots
        my_color[1] = "#cadfd9"
            
        pt = ggplot(cur_df, aes(x=x, y=y, group=factor(cur_df$label))) +
            geom_point(aes(colour = factor(cur_df$label)), stroke=0,
                       size=2, show.legend=TRUE) +
            scale_color_manual(values=my_color) +
            labs(color='Well') +
            theme(panel.background = element_blank(),
                  legend.key.height=unit(1, "line"),
                  plot.title = element_text(
                      family = "San Francisco Display Regular",
                      hjust = 0.5, size=15
                      ),
                  legend.position="bottom") +
            scale_alpha_continuous(guide = FALSE) +
            # Set axis limit to make all plots have the same scale
            xlim(x_low, x_high) +
            ylim(y_low, y_high) +
            ggtitle(sprintf("t-SNE plot %d of Plate %s", i, channel))
    
        # Save the image
        ggsave(sprintf("t-sne_%s_%d.png", channel, i), pt,
               width = 7, height = 7, units=("in"), limitsize = F)
    }
}


# Split the t-sne data of one plate into n subsets with size fold_size
# Each subset have all the DMSO data in this plate
# Subset is assigned with colors
get_json = function(df, channel, fold_size=30, dim=3){
    x = split(df, df$label)
    
    dmso_df = x[[sprintf("%s_DMSO", channel)]]
    x[sprintf("%s_DMSO", channel)] = NULL
    
    # Add a fixed color for DMSO, and other disinctive colors for others
    dmso_df$color = rep("#cadfd9", dim(dmso_df)[1])
    my_color = distinctColorPalette(fold_size)
    
    # Fill the list to encode JSON
    results = list()
    
    for (i in 1:ceiling(length(x) / fold_size)){
        print(sprintf("Encoding %d of %s", i, channel))
        upper = min(i * fold_size, length(x))
        
        # Preprocess the x list to add a color column
        count = 1
        for (t in ((i - 1) * fold_size + 1) : upper){
            x[[t]]$color = rep(my_color[count], dim(x[[t]])[1])
            count = count + 1
        }
        cur_df = rbind(dmso_df, x[[(i - 1) * fold_size + 1]])
        for (t in ((i - 1) * fold_size + 2) : upper){
            cur_df = rbind(cur_df, x[[t]])
        }
        
        # Convert the dfs to a named list
        cur_list = list()
        cur_list[["x"]] = cur_df$x
        cur_list[["y"]] = cur_df$y
        cur_list[["z"]] = cur_df$z
        cur_list[["label"]] = cur_df$label
        cur_list[["color"]] = cur_df$color
        results[[i]] = cur_list
    }
    # Save the json file
    write(toJSON(results, pretty=T), file=sprintf("%s.json", channel))
}

df = read.csv("tsne_perplexity_50.csv")

# Fix the appropriate scale fro all plots
x_low = min(df$x) - 10
x_high = max(df$x) + 10
y_low = min(df$y) - 10
y_high = max(df$y) + 10

# Get the total channels in the csv file
channels = sort(unique(gsub("()_.*", "\\1", df$label)))

# Iterate through all channels and plot subplots of each of them
for (channel in channels){
    channel_df = df[grep(channel, df$label),]
    # R requires to refactor after subsetting
    channel_df$label = factor(channel_df$label)

    # plot_plate(channel_df, channel, x_low, x_high, y_low, y_high)
    # get_json(channel_df, channel)
}
