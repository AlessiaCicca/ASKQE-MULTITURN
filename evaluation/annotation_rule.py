import json


input_file = "/content/ASKQE-MULTITURN/results/biomqm/reference_src_mt_perturb.jsonl"
output_file = "/content/ASKQE-MULTITURN/results/biomqm/human_ratings.jsonl"


with open(input_file, "r", encoding="utf-8") as infile, open(output_file, "w", encoding="utf-8") as outfile:
    for line in infile:
        data = json.loads(line.strip())
        
        # Controlla la lista degli errori nel campo 'target_errors'
        reject_decision = False
        if "target_errors" in data:
            for error in data["target_errors"]:
                # Se uno degli errori ha gravità "critical" o "major", imposta la decisione su "reject"
                if error.get("severity", "").lower() in ["critical", "major"]:
                    reject_decision = True
                    break
        
        # Imposta la decisione basata sulla gravità dell'errore
        if reject_decision:
            data["decision"] = "reject"
        else:
            data["decision"] = "accept"
        
        # Scrivi l'entry modificata nel file di output
        json.dump(data, outfile, ensure_ascii=False)
        outfile.write("\n")
