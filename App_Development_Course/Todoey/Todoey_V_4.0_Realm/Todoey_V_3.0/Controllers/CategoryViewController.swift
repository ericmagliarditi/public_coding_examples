//
//  CategoryViewController.swift
//  Todoey_V_3.0
//
//  Created by Eric Magliarditi on 6/17/19.
//  Copyright © 2019 Eric Magliarditi. All rights reserved.
//

import UIKit
import RealmSwift

class CategoryViewController: SwipeTableViewController {
    
    var categoryArray: Results<Category>?
    
    let realm = try! Realm()
    //Use the ! here so we can hint at bad code or it can be used as something to force the issue at hand so the compiler runs it

    override func viewDidLoad() {
        super.viewDidLoad()
        
        loadCategories()
        
    }

    // MARK: - Table view data source
    override func tableView(_ tableView: UITableView, numberOfRowsInSection section: Int) -> Int {
        // #warning Incomplete implementation, return the number of rows
        /**
         Below is the nil collesing operator - says the left hand side maybe nil and if it is nil assign it to 1, if it is not nil then get the true count
         A single questionmark says do this if not nil
        */
        return categoryArray?.count ?? 1
    }
    
    override func tableView(_ tableView: UITableView, cellForRowAt indexPath: IndexPath) -> UITableViewCell {
        
        /**
         This allows us to tap into the cell at the super view, i.e the inhereted class but then we can adjust it accordingly
        */
        
        let cell = super.tableView(tableView, cellForRowAt: indexPath)
        
        cell.textLabel?.text = categoryArray?[indexPath.row].name ?? "No Categories Added"
        
        return cell
    }
    
    //MARK: - Table View Delegate Methods
    
    override func tableView(_ tableView: UITableView, didSelectRowAt indexPath: IndexPath) {
        performSegue(withIdentifier: "goToItems", sender: self)
    }
    
    override func prepare(for segue: UIStoryboardSegue, sender: Any?) {
        
        let destinationVC = segue.destination as! ToDoListViewController
        
        if let indexPath = tableView.indexPathForSelectedRow {
            destinationVC.selectedCategory = categoryArray?[indexPath.row]
        }
    }
    
//    override func tableView(_ tableView: UITableView, cellForRowAt indexPath: IndexPath) -> UITableViewCell {
//        let cell = tableView.dequeueReusableCell(withIdentifier: "Cell") as! SwipeTableViewCell
//        cell.delegate = self
//        return cell
//    }
    
    //MARK: - Data Manipulation Methods
    
    func saveCategories(category: Category) {
        do {
            try realm.write {
                realm.add(category)
            }
        }
        catch {
            print("Error saving category data \(error)")
        }
        
        tableView.reloadData()
    }
    
    func loadCategories() {
        
        /**
         To read in data - we dont need a fetch request like in CoreData
         We can use this one line of code to grab the categories
         Cant just assing it to the categoryArray
         The datatybe of the objects we are getting back is Results which is a container type from Realm Swift
         Thus you cant assign it simply to the category Array
        */
        categoryArray = realm.objects(Category.self)
        
        tableView.reloadData()
    }
    
    //Mark Delete Data
    override func updateModel(at indexPath: IndexPath) {
        if let categoryForDeletion = self.categoryArray?[indexPath.row] {
            do {
                try self.realm.write {
                    self.realm.delete(categoryForDeletion)
                }
            }
            catch {
                print("Error deleting category \(error)")
            }
        }
    }
    
    //MARK: Add Category Methods
    
    @IBAction func addButtonPressed(_ sender: UIBarButtonItem) {
        
        var textField = UITextField()
        
        let alert = UIAlertController(title: "Add New Category", message: "", preferredStyle: .alert)
        
        let action = UIAlertAction(title: "Add", style: .default) { (action) in
            
            let newCategory = Category()
            
            newCategory.name = textField.text!
            
//            self.categoryArray.append(newCategory)
            //It automatically updates so it doesnt need the append
            
            self.saveCategories(category: newCategory)
            
        }
        
        alert.addAction(action)
        
        alert.addTextField { (field) in
            textField = field
            textField.placeholder = "Add a new Category"
        }
        
        present(alert, animated: true, completion: nil)
        
        
    }
    
}
