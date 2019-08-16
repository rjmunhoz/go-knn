package main

import (
	"encoding/csv"
	"fmt"
	"log"
	"math"
	"os"
	"sort"
	"strconv"
)

type neighbour struct {
	distance float64
	class    string
}

func readFile(path string) ([][]string, error) {
	file, err := os.Open(path)

	if err != nil {
		return [][]string{}, err
	}

	defer file.Close()

	reader := csv.NewReader(file)
	return reader.ReadAll()
}

func getColumn(records [][]string, desiredColumn int) []string {
	var column []string

	for _, record := range records {
		column = append(column, record[desiredColumn])
	}

	return column
}

func distinct(records []string) []string {
	found := map[string]bool{}
	var result []string

	for _, record := range records {
		if found[record] == false {
			found[record] = true
			result = append(result, record)
		}
	}

	return result
}

func last(attributes []string) string {
	return attributes[len(attributes)-1]
}

func getClassInstaces(instances [][]string, class string) [][]string {
	classInstances := make([][]string, 0)

	for _, instance := range instances {
		if last(instance) == class {
			classInstances = append(classInstances, instance)
		}
	}

	return classInstances
}

func getEuclidianDistance(classifiedInstance []string, unclassifiedInstance []string) float64 {
	// We need to remove the "class" column
	classified := classifiedInstance[:len(classifiedInstance)-1]
	unclassified := unclassifiedInstance[:len(unclassifiedInstance)-1]

	sum := 0.0

	for attributeIndex := range classified {
		classifiedAttribute, _ := strconv.ParseFloat(classified[attributeIndex], 32)
		unclassifiedAttribute, _ := strconv.ParseFloat(unclassified[attributeIndex], 32)
		sum += math.Pow(classifiedAttribute-unclassifiedAttribute, 2)
	}

	return math.Sqrt(sum)
}

func getMostCommonClass(knn []neighbour) string {
	classesCount := make(map[string]int)

	for _, neighbour := range knn {
		classesCount[neighbour.class]++
	}

	var result string
	var lastCount int

	for class, count := range classesCount {
		if count > lastCount {
			lastCount = count
			result = class
		}
	}

	return result
}

func classify(dataset [][]string, unclassifiedInstance []string, k int) string {
	var neighbours []neighbour

	for _, instance := range dataset {
		neighbours = append(neighbours, neighbour{
			distance: getEuclidianDistance(instance, unclassifiedInstance),
			class:    instance[len(instance)-1],
		})
	}

	sort.Slice(neighbours, func(i, j int) bool {
		return neighbours[i].distance < neighbours[j].distance
	})

	knn := neighbours[:k-1]

	return getMostCommonClass(knn)
}

func main() {
	records, err := readFile("data/dataset.csv")

	if err != nil {
		log.Fatalln("Error: ", err)
	}

	columnCount := len(records[0])

	classes := distinct(getColumn(records, columnCount-1))

	var training [][]string
	var testing [][]string

	var testingPercentage = 0.8

	for _, class := range classes {
		values := getClassInstaces(records, class)
		valuesCount := len(values)
		testingValuesCount := math.Floor(float64(valuesCount) * testingPercentage)

		testing = append(testing, values[:int(testingValuesCount)]...)
		training = append(training, values[int(testingValuesCount):]...)
	}

	hits := 0

	for _, testInstance := range testing {
		actualClass := last(testInstance)
		predictedClass := classify(training, testInstance, 10)
		
		if (actualClass == predictedClass) {
			hits++
			continue
		}

		fmt.Printf("Miss! Tumor %s classificado como %s\n", actualClass, predictedClass)
	}

	hitPercentage := (hits * 100) / len(testing)

	fmt.Printf("total: %d, training: %d, testing: %d, hits: %d, accuracy: %d%%\n", len(records), len(training), len(testing), hits, hitPercentage)
}
