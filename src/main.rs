use rand::seq::SliceRandom;
use std::env;
use std::fs::File;
use std::io::prelude::*;
use std::io::BufReader;

#[derive(Debug)]
struct Arguments {
    in_file: String,
    p0: u32,
}

fn parse_args() -> Arguments {
    let args: Vec<String> = env::args().skip(1).collect();

    Arguments {
        in_file: args[0].clone(),
        p0: args[1].clone().parse().unwrap(),
    }
}
fn parse_graph(f: File) -> (Vec<(u32, u32)>, Vec<Vec<u32>>) {
    let mut reader = BufReader::new(f);
    let mut first_line: String = "".to_string();
    let _i = reader.read_line(&mut first_line);
    first_line = first_line.strip_suffix("\n").unwrap().to_owned();
    let split_first = first_line
        .split(" ")
        .map(|x| x.parse::<u32>().unwrap())
        .collect::<Vec<_>>();
    println!("{:?}", split_first);
    println!("{}", &first_line);
    let mut adj_list: Vec<Vec<u32>> = Vec::<Vec<u32>>::with_capacity(split_first[0] as usize);
    let mut edg_list: Vec<(u32, u32)> = vec![];
    for line_ in reader.lines().enumerate() {
        let line = line_.1.unwrap();
        let line: Vec<u32> = line.split(" ").map(|x| x.parse::<u32>().unwrap()).collect();

        let idx = line_.0;
        if line.len() > 0 {
            adj_list.push(vec![]);
            for numb in &line {
                adj_list[idx].push(*numb);
                if *numb > idx as u32 {
                    edg_list.push((idx as u32, *numb));
                }
            }
        }
        // println!("{:?}", line);
        // println!("{:?}", idx);
    }
    // println!("{:?}", adj_list);
    // println!("{:?}", edg_list);
    (edg_list, adj_list)
}

#[derive(Clone, Debug)]
struct Individual {
    chromosome: Vec<u32>,
    fitness: u32,
    lockdown_edgelist: Vec<(u32, u32)>,
    lockdown_adjlist: Vec<Vec<u32>>,
}

impl Individual {
    fn new() -> Self {
        todo!()
    }
}
struct Population {
    pop: Vec<Individual>,
}

impl Population {
    fn init_pop(
        percent_lock: f32,
        pop_size: i32,
        edg_list: Vec<(u32, u32)>,
        adj_list: Vec<u32>,
    ) -> Self {
        let edg_count = edg_list.len();
        let mut pop_vec: Vec<u32> = Vec::with_capacity(pop_size as usize);
        let num_remove = (percent_lock * edg_count as f32) as usize;
        let mut init_list = vec![0; edg_count];
        for (i, v) in init_list.iter_mut().enumerate() {
            *v = i as u32;
        }
        let mut rng = &mut rand::thread_rng();
        for i in 0..pop_size {
            let mut new_ind: Vec<u32> = vec![1; edg_count];
            let rindices: Vec<u32> = init_list
                .choose_multiple(&mut rng, num_remove)
                .cloned()
                .collect();
            for j in &rindices {
                new_ind[*j as usize] = 0;
            }
        }

        todo!()
    }
}

fn main() {
    let args = parse_args();
    println!("{}", args.p0);
    let f = File::open(&args.in_file).unwrap();

    let (elist, alist) = parse_graph(f);
    println!("{:?}", alist);
    // println!("{:?}", elist);
}
