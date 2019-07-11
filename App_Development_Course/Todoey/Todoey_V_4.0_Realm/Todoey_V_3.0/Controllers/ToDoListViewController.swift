//
//  ViewController.swift
//  Todoey_V_3.0
//
//  Created by Eric Magliarditi on 6/17/19.
//  Copyright © 2019 Eric Magliarditi. All rights reserved.
//

import UIKit
import RealmSwift
import ChameleonFramework

class ToDoListViewController: SwipeTableViewController {
    
    var todoItems: Results<Item>?
    let realm = try! Realm()
    
    
    @IBOutlet weak var searchBar: UISearchBar!
    
    
    var selectedCategory : Category? {
        didSet {
            loadItems()
        }
    }
    
    override func viewDidLoad() {
        super.viewDidLoad()
        
//        print(FileManager.default.urls(for: .documentDirectory, in: .userDomainMask))
        
        tableView.separatorStyle = .none
        
        
        

    }
    
    /**
     Here is the issue - view did load loads up the items before the navigation bar is accurately accessed by the controller - so we use this viewWillAppear which gets called right after viewdid load in order to set the navigation controller!
    */
    override func viewWillAppear(_ animated: Bool) {
        
        //Remember we are in a optional binding so can make it an absolute reference
        title = selectedCategory!.name
    
        guard let colorHex = selectedCategory?.colorCat else{fatalError()}
            //Use guard when we dont have an else in the if let especially when the app will not work if the if let fails
        
        updateNavBar(withHexCode: colorHex)
       
        
    }
    
    /**
     Ok so we also need to call something when the nav bar is just about to disappear because it carries over the color which we dont want
     This is the point where we are dismssing this
    */
    override func viewWillDisappear(_ animated: Bool) {
     
        updateNavBar(withHexCode: "434343")
    }
    
    /**
     We are going to create a function that deals with nav Bar since there is too much code repeating
    */
    //MARK: - NavBar Setup Code Methods
    func updateNavBar(withHexCode colorHexCode: String){
        guard let navBar = navigationController?.navigationBar else {fatalError("Nav Bar Not loaded yet")}
        
        //This will change the buttons in the navBar
        guard let navBarColor = UIColor(hexString: colorHexCode) else {fatalError()}
        
        navBar.tintColor = ContrastColorOf(navBarColor, returnFlat: true)
        
        searchBar.barTintColor = navBarColor
        
        navBar.barTintColor = navBarColor
        
        //Must use larget title since that is what we are using
        navBar.largeTitleTextAttributes = [NSAttributedString.Key.foregroundColor : ContrastColorOf(navBarColor, returnFlat: true)]
        
    }
    
    
    //MARK: - TableView DataSource Methods
    
    override func tableView(_ tableView: UITableView, numberOfRowsInSection section: Int) -> Int {
        return todoItems?.count ?? 1
    }
    
    override func tableView(_ tableView: UITableView, cellForRowAt indexPath: IndexPath) -> UITableViewCell {
        
        let cell = super.tableView(tableView, cellForRowAt: indexPath)
        
        if let item = todoItems?[indexPath.row] {

            cell.textLabel?.text = item.title

            //Ternary Operations
            // value = condition ? valueIfTrue : valueIfFalse
            cell.accessoryType = item.done ? .checkmark : .none
            
            /**
             We use optional binding here but notice we can use the exclamation point on the todoItems since we already are in an optional binding of the todoItems - meaning it can never be nil within this code block
            */
            
            
            if let colorFloat = UIColor(hexString: selectedCategory!.colorCat )?.darken(byPercentage: CGFloat(indexPath.row) / CGFloat(todoItems!.count)) {
                
                cell.backgroundColor = colorFloat
                
                cell.textLabel?.textColor = ContrastColorOf(colorFloat, returnFlat: true)
                }
        }
        else {
            cell.textLabel?.text = "No Items Added"
        }
        
        return cell
    }
    
    //MARK: - TableView Delegate Methods
    
    override func tableView(_ tableView: UITableView, didSelectRowAt indexPath: IndexPath) {
        
        /**
         This is where we update data using Realm!
         Also note that we keep using the if let as a means to say that if it is not nil!
         Use realm.delete if we want to delete it
        */
        if let item = todoItems?[indexPath.row] {
            do {
                try realm.write {
                    item.done = !item.done
//                    realm.delete(item)
                }
            }
            catch {
                print("Error updating data \(error)")
            }
           
        }
        tableView.reloadData()
        
        tableView.deselectRow(at: indexPath, animated: true)
    }
    
    
    //Mark Delete Data
    override func updateModel(at indexPath: IndexPath) {
        if let itemForDeletion = self.todoItems?[indexPath.row] {
            do {
                try self.realm.write {
                    self.realm.delete(itemForDeletion)
                }
            }
            catch {
                print("Error deleting category \(error)")
            }
        }
    }
    
    
    @IBAction func addButtonPressed(_ sender: UIBarButtonItem) {
       
        var textField = UITextField()
        
        let alert = UIAlertController(title: "Add New Item", message: "", preferredStyle: .alert)
        
        let action = UIAlertAction(title: "Add Item", style: .default) { (action) in
            
            if let currentCategory = self.selectedCategory {
                
                do {
                    try self.realm.write {
                        let newItem = Item()
                        newItem.title = textField.text!
                        newItem.dateCreated = Date()
                        currentCategory.items.append(newItem)
                        self.realm.add(newItem)
                    }
                }
                catch{
                    print("Error \(error)")
                }
 
            }
            self.tableView.reloadData()
        }
        
        alert.addTextField { (alertTextField) in
            alertTextField.placeholder = "Create New Items"
            textField = alertTextField
        }
        
        alert.addAction(action)
        
        present(alert, animated: true, completion: nil)
    }
    
    //MARK: - Model Manipulation Methods
    
    
    func loadItems(){
        
        todoItems = selectedCategory?.items.sorted(byKeyPath: "dateCreated", ascending: true)
        
        tableView.reloadData()
        
    }

}

//MARK: Search Bar Methods
extension ToDoListViewController: UISearchBarDelegate {

    func searchBarSearchButtonClicked(_ searchBar: UISearchBar) {
        /**
         We filter by the SQL type query and filter based on date created
        */
        todoItems = todoItems?.filter("title CONTAINS[cd] %@", searchBar.text!).sorted(byKeyPath: "dateCreated", ascending: false)
        
        tableView.reloadData()
    }

    func searchBar(_ searchBar: UISearchBar, textDidChange searchText: String) {
        if searchBar.text?.count == 0 {
            /**
             This is when we basically want to get rid of the search box
            */
            loadItems()

            DispatchQueue.main.async {
                searchBar.resignFirstResponder()
            }
        }

    }
}
