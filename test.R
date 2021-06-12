
filter_year <- function(df, year){
  df_1 <- df[df$year == year, ]
  df_2 <- df_1[order(-df_1$popularity), ]
  df_3 <- df_2[1:50, ]
  return(df_3)
}

get_three <- function(df){
  one <- df[1,]
  two <- df[24,]
  three <- df[50,]
  three_df <- bind_rows(one, two, three)
  return (three_df)
}

library(ggradar)
library(dplyr)
library(scales)
library(tibble)
library(tidyverse)
library(lazyeval)

radar <- function (df){
  df_radar <- df %>% 
    as_tibble(rownames = 'group') %>% 
    mutate_at(vars(-group), rescale) %>% 
    tail(4) %>% 
    select(1:10)
  ggradar(df_radar)
}

