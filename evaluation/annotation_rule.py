import json

#This script reads a JSONL file containing human-annotated translation data and assigns a 
#binary decision label (“accept” or “reject”) to each entry based on the severity of target-side 
#errors. If any error in target_errors has severity marked as major or critical, the translation 
#is labeled as “reject”; otherwise, it is labeled as “accept”, and the updated records are written 
#to a new JSONL output file.

input_file = "/content/ASKQE-MULTITURN/results/biomqm/reference_src_mt_perturb.jsonl"
output_file = "/content/ASKQE-MULTITURN/results/biomqm/human_ratings.jsonl"


with open(input_file, "r", encoding="utf-8") as infile, open(output_file, "w", encoding="utf-8") as outfile:
    for line in infile:
        data = json.loads(line.strip())
        reject_decision = False
        if "target_errors" in data:
            for error in data["target_errors"]:
                if error.get("severity", "").lower() in ["critical", "major"]:
                    reject_decision = True
                    break
        
        if reject_decision:
            data["decision"] = "reject"
        else:
            data["decision"] = "accept"
       
        json.dump(data, outfile, ensure_ascii=False)
        outfile.write("\n")
