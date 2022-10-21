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
    let split_first = first_line.split(" ").collect::<Vec<_>>();
    println!("{:?}", split_first);
    println!("{}", &first_line);
    let mut adj_list: Vec<Vec<u32>> =
        Vec::<Vec<u32>>::with_capacity(split_first[0].parse().unwrap());
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

fn main() {
    let args = parse_args();
    println!("{}", args.p0);
    let f = File::open(&args.in_file).unwrap();

    let (elist, alist) = parse_graph(f);
    println!("{:?}", alist);
    // println!("{:?}", elist);
}
